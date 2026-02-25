from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.utils import get_atoms_from_template_db, get_calculator, get_features_bulk_and_gas
from gen_catalyst_design.db import Database
from ase.io import read
from ase.db import connect
from ase_ml_models.databases import get_atoms_list_from_db
from gen_catalyst_design.db import Database
from catalyst_opt_tools.utilities import preprocess_features, update_atoms_list
import torch
import yaml
import os




def main():
    miller_index = "100"
    universal_pth_header = "../../.."
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
        mechanism_pth_header=os.path.join(universal_pth_header,"yaml_files/reaction_mechanism"),
        features_bulk=features_bulk,
        features_gas=features_gas
    )

    reaction_mechanism.set_template_atoms_list(
        db_filename=f"{miller_index}_templates.db",
        pth_header=f"../../../databases/cluster_templates/{miller_index}"
    )
    
    models = [
        "../../active_learning/iter_0",
        "../../active_learning/iter_1"
    ]
    temps = [1.0]
    for model in models:
        for temp in temps:
            filename = os.path.join(model, f"samples_{temp}.traj")
            atoms_list = read(filename=filename, index=load_indices)
            filtered_atoms_list = [atoms for atoms in atoms_list if "O" not in atoms.symbols]
            score_dicts = []
            elements_list = []
            for atoms in filtered_atoms_list:
                elements = atoms.get_chemical_symbols()
                score_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
                    symbols=elements
                )
                score_dicts.append(score_dict)
                elements_list.append(elements)
            
            database = Database.establish_connection(
                filename=f"rate_evals_{temp}.db",
                miller_index="100",
                pth_header=model
            )
            data_dicts = []
            for elements, score_dict in zip(elements_list, score_dicts):
                data_dicts.append({"elements":elements, "score_dict":score_dict, "batch":0})
            database.write_data_to_tables(data_dicts=data_dicts, append=False)
            database.close_connection()
            print(f"done evaluating: model {model}, temp: {temp}")



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