# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------
import random
import numpy as np
import sys
sys.path.insert(0, "../")
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.utils import get_atoms_from_template_db, get_calculator, get_features_bulk_and_gas, get_full_element_pool
from gen_catalyst_design.db import Database
import torch
from ase.io import read


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    model = "WWL-GPR"
    template_type = "surface" # cluster | surface
    miller_index = "100" # 100 | 111
    elements = get_full_element_pool() # Elements of the surface.
    random_seed = 42 # Random seed for reproducibility.
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    
    atoms_traj_file = None
    n_samples = 4
    # Get features.
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header="../yaml_files/features")
    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT_all.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header="../databases/DFT_database",
         train_kwargs=train_kwargs
    )
    
    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"../databases/{template_type}_templates"
    )
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        template_atoms_list=template_atoms_list,
        calculator=calculator,
        features_bulk=features_bulk,
        features_gas=features_gas,
        mechanism_pth_header="../yaml_files/reaction_mechanism"
    )

    database = Database.establish_connection(
        filename="test_pred.db",
        database_kwargs={"template_atoms_surf":template_atoms_list[0], "append":False},
    )

    data_dicts = []
    if atoms_traj_file is None:
        for _ in range(n_samples):
            symbols = random.choices(population=elements, k=n_atoms_surf)
            result_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
                symbols=symbols
            )
            data_dict = {"elements":symbols, "gen_iter":1, "score_dict":result_dict}
            data_dicts.append(data_dict)
            print(f"Symbols =", ",".join(symbols))
            rate = result_dict["rate"]
            print(f"Reaction rate = {rate:+7.3e} [1/s]")
            #database.write_data_to_tables(data_dicts=data_dicts)
    else:
        atoms_list = read(filename=atoms_traj_file, index=":")
        for atoms in atoms_list:
            symbols = atoms.get_chemical_symbols()
            result_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
                symbols=symbols
            )
            data_dict = {"elements":symbols, "gen_iter":1, "score_dict":result_dict}
            data_dicts.append(data_dict)
            print(f"Symbols =", ",".join(symbols))
            rate = result_dict["rate"]
            print(f"Reaction rate = {rate:+7.3e} [1/s]")
    database.write_data_to_tables(data_dicts=data_dicts)
    #rates = [datadict["score_dict"]["rate"] for datadict in data_dicts]





# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()