# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import random
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI
import os
import sys
sys.path.insert(0, "../")
from catalyst_opt_tools.utilities import update_atoms_list, preprocess_features
from gen_catalyst_toolkit.calculators import GraphCalculator, GCNNCalculator
from gen_catalyst_toolkit.reaction_rates import ReactionMechanism
from ase_ml_models.databases import get_atoms_list_from_db
from gen_catalyst_toolkit.db import Database
from ase.db import connect
import torch
from ase.io import read, write


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    model = "WWL-GPR"
    # Parameters.
    miller_index = "100" # 100 | 111
    elements = ["Rh", "Cu", "Au"] # Elements of the surface.
    random_seed = 42 # Random seed for reproducibility.
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    
    atoms_traj_file = None
    n_samples = 4
    # Get features.
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header="yaml_files/features")
    
    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header="databases/DFT_database",
         train_kwargs=train_kwargs
    )
    
    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"databases/templates/{miller_index}"
    )
    
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        calculator=calculator,
        mechanism_pth_header="yaml_files/reaction_mechanism"
    )

    data_dicts = []
    if atoms_traj_file is None:
        for _ in range(n_samples):
            symbols = random.choices(population=elements, k=n_atoms_surf)
            result_dict = reaction_rate_of_RDS_from_symbols(
                reaction_mechanism=reaction_mechanism,
                symbols=symbols,
                template_atoms_list=template_atoms_list, 
                features_bulk=features_bulk, 
                features_gas=features_gas, 
                n_atoms_surf=n_atoms_surf
            )
            data_dict = {"elements":symbols, "batch":1, "score_dict":result_dict}
            data_dicts.append(data_dict)
            print(f"Symbols =", ",".join(symbols))
            rate = result_dict["rate"]
            print(f"Reaction rate = {rate:+7.3e} [1/s]")
    else:
        atoms_list = read(filename=atoms_traj_file, index=":")
        for atoms in atoms_list:
            symbols = atoms.get_chemical_symbols()
            result_dict = reaction_rate_of_RDS_from_symbols(
                reaction_mechanism=reaction_mechanism,
                symbols=symbols,
                template_atoms_list=template_atoms_list, 
                features_bulk=features_bulk, 
                features_gas=features_gas, 
                n_atoms_surf=n_atoms_surf
            )
            data_dict = {"elements":symbols, "batch":1, "score_dict":result_dict}
            data_dicts.append(data_dict)
            print(f"Symbols =", ",".join(symbols))
            rate = result_dict["rate"]
            print(f"Reaction rate = {rate:+7.3e} [1/s]")
    #rates = [datadict["score_dict"]["rate"] for datadict in data_dicts]

    database = Database.establish_connection(filename="diffusion_model/test_pred5.db", miller_index=miller_index)
    database.write_data_to_tables(data_dicts=data_dicts, append=False)

# -------------------------------------------------------------------------------------
# GET RATE FROM SYMBOLS
# -------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------
# GET FEATURES
# -------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------
# GET TEMPLATE ATOMS
# -------------------------------------------------------------------------------------

def get_atoms_from_template_db(
    db_filename:str,
    pth_header:str=None
):
    """
    Get atoms from template database.
    """
    # Read atoms objects from templates database.
    if pth_header is not None:
        db_filename = os.path.join(pth_header, db_filename)
    db_ase = connect(db_filename)
    atoms_list = get_atoms_list_from_db(db_ase=db_ase)
    # Get number of atoms in the surface.
    n_atoms_surf = len([
        atoms for atoms in atoms_list if atoms.info["species"] == "clean"
    ][0])
    # Return the list of atoms objects.
    return atoms_list, n_atoms_surf

# -------------------------------------------------------------------------------------
# GET CALCULATOR
# -------------------------------------------------------------------------------------

def get_calculator(model, miller_index):
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


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()