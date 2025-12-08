from gen_catalyst_design.reaction_rates import ReactionMechanism
from ase.io import read
from ase.db import connect
from ase_ml_models.databases import get_atoms_list_from_db
from gen_catalyst_design.db import Database
from catalyst_opt_tools.utilities import preprocess_features, update_atoms_list
import torch
import yaml
import os




def main():
    opt_methods = [
        #"random_search",
        "GeneticAlgorithm"
    ]
    
    db = connect("../../databases/templates/100/100_templates.db")
    template_atoms_list = get_atoms_list_from_db(db_ase=db)
    n_atoms_surf = len(template_atoms_list[0])
    miller_index = "100"
    
    universal_pth_header = "../.."
    load_indices = ":"
    model = "WWL-GPR"

    features_bulk, features_gas = get_features_bulk_and_gas(pth_header=os.path.join(universal_pth_header, "yaml_files/features"))
    #Get calculator of model type and training parameter

    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    calculator.train_model_from_db(
        db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
        features_bulk=features_bulk, 
        features_gas=features_gas, 
        db_pth_header=os.path.join(universal_pth_header,"databases/DFT_database"),
        train_kwargs=train_kwargs
    )

    reaction_mechanism = ReactionMechanism(
        calculator=calculator,
        mechanism_pth_header=os.path.join(universal_pth_header,"yaml_files/reaction_mechanism")
    )
    
    models = [
        "model_003"
    ]

    for opt_method in opt_methods:
        file_dir = os.path.join("samples", opt_method)
        #models = os.listdir(file_dir)
        for model in models:
            class_divisions = ["class_7"]#os.listdir(os.path.join(file_dir, model))
            for i, cls in enumerate(class_divisions):
                samples_files = ["g_0.8_scale.traj"]#os.listdir(os.path.join(file_dir, model, cls))
                for sample_file in samples_files:
                    filename = os.path.join(file_dir, model, cls, sample_file)
                    atoms_list = read(filename=filename, index=load_indices)
                    elements_list = [atoms.get_chemical_symbols() for atoms in atoms_list]
                    guidance_scale = split_traj_file(filename=sample_file)
                    score_dicts = reaction_rate_calculation(
                        symbols_list=elements_list,
                        template_atoms_list=template_atoms_list,
                        n_atoms_surf=n_atoms_surf,
                        reaction_mechanism=reaction_mechanism,
                        features_bulk=features_bulk,
                        features_gas=features_gas
                    )
                    db_pth_header = os.path.join(file_dir, model, "rate_evals", cls)
                    if not os.path.exists(db_pth_header):
                        os.makedirs(db_pth_header)
                    database = Database.establish_connection(
                        filename=f"g_{guidance_scale}_scale.db",
                        miller_index="100",
                        pth_header=db_pth_header
                    )
                    data_dicts = []
                    for elements, score_dict in zip(elements_list, score_dicts):
                        data_dicts.append({"elements":elements, "score_dict":score_dict, "batch":0})
                    database.write_data_to_tables(data_dicts=data_dicts, append=False)
                    database.close_connection()
                    print(f"done evaluating: method: {opt_method}, class: {cls}, guidance_scale: {guidance_scale}")


def get_features_bulk_and_gas(
        bulk_filename:str="features_bulk.yaml", 
        gas_filename:str="features_gas.yaml", 
        pth_header:str=None
        ) -> tuple:
        """
        Get features for bulk and gas phase.
        """
        if pth_header is not None:
            bulk_filename = os.path.join(pth_header, bulk_filename)
            gas_filename = os.path.join(pth_header, gas_filename)
        # Read features from yaml files.
        with open(bulk_filename, "r") as fileobj:
            features_bulk = yaml.safe_load(fileobj)

        with open(gas_filename, "r") as fileobj:
            features_gas = yaml.safe_load(fileobj)
        # Preprocess features.
        features_bulk = preprocess_features(features_dict=features_bulk)
        features_gas = preprocess_features(features_dict=features_gas)
        # Return parameters.
        return features_bulk, features_gas

def get_calculator(model, miller_index):
    from gen_catalyst_design.calculators import GCNNCalculator, GraphCalculator
    if model == "WWL-GPR":
        calculator = GraphCalculator(
             miller_index=miller_index,
             kernel="GPR"
        )
        train_kwargs = {}
    elif model == "GCNN":
        calculator = GCNNCalculator(
            miller_index=miller_index
        )
        network_hyperparams = {"hidden_dim":128,
            "n_conv_layers": 4,
            "n_lin_layers": 2,
            "conv_type": "ARMAConv",
            "dropout":0.0,
            "activation": torch.nn.functional.relu,
            #"use_batch_norm":False,
            #"use_residual":False
        }
        train_kwargs = {
                    "early_stop":True,
                    "num_epochs":500,
                    "lr": 1e-4,
                    "batch_size": 16,
                    "target": "E_form",
                    "val_split": 0.1,
                    "early_stopping_patience": 100,
                    "early_stopping_delta": 1e-4,
                    "weight_decay":0.0,
                    "hyperparams":network_hyperparams
        }
    else:
         raise Exception(f"calculator of type: {model} has not been implemented yet")
    return calculator, train_kwargs


def reaction_rate_of_RDS_from_symbols(
    reaction_mechanism:ReactionMechanism,
    symbols: list,
    template_atoms_list: list,
    features_bulk: dict,
    features_gas: dict,
    n_atoms_surf: int,
):
    """
    Get reaction rate of the RDS from the surface symbols.
    """
    # Update elements and features of atoms for predictions.
    update_atoms_list(
        atoms_list=template_atoms_list,
        features_bulk=features_bulk,
        features_gas=features_gas,
        symbols=symbols,
        n_atoms_surf=n_atoms_surf,
    )
    # Predict formation energies with a calculator.
    # Calculate reaction rate from reaction mechanism.
    score_dict = reaction_mechanism(atoms_list=template_atoms_list)
    return score_dict


def split_traj_file(filename):
    splitted_name = filename.split("_")
    return splitted_name[1]



def reaction_rate_calculation(
        symbols_list:list,
        template_atoms_list:list,
        n_atoms_surf:int,
        reaction_mechanism,
        features_gas,
        features_bulk
    ):
   
    score_dicts = []
    for symbols in symbols_list:
        score_dict = reaction_rate_of_RDS_from_symbols(
            reaction_mechanism=reaction_mechanism,
            symbols=symbols,
            template_atoms_list=template_atoms_list,
            features_bulk=features_bulk,
            features_gas=features_gas,
            n_atoms_surf=n_atoms_surf
        )
        score_dicts.append(score_dict)
    return score_dicts


def filter_data_dicts(data_dicts:list, rate_min:float=None):
    if rate_min is None:
        return data_dicts
    else:
        filtered_datadicts = []
        for datadict in data_dicts:
            if datadict["rate"] < rate_min:
                pass
            else:
                filtered_datadicts.append(datadict)
        return filtered_datadicts



if __name__ == "__main__":
    main()