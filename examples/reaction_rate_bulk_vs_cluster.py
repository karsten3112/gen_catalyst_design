

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------
import random
import numpy as np
import sys
sys.path.insert(0, "../")
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.utils import get_atoms_from_template_db, get_calculator, get_features_bulk_and_gas
from gen_catalyst_design.db import Database
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
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header="../yaml_files/features")
    
    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header="../databases/DFT_database",
         train_kwargs=train_kwargs
    )
    
    reaction_mechanism = setup_reaction_mechanism(
        miller_index=miller_index,
        surface_type="bulk",
        calculator=calculator,
        features_bulk=features_bulk,
        features_gas=features_gas
    )

    data_dicts = []
    print("EVALUATING RATE FOR RANDOM BULK-SURFACES")
    for _ in range(n_samples):
        symbols = random.choices(population=elements, k=reaction_mechanism.n_atoms_surf)
        result_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
            symbols=symbols
        )
        data_dict = {"elements":symbols, "batch":1, "score_dict":result_dict}
        data_dicts.append(data_dict)
        print(f"Symbols =", ",".join(symbols))
        rate = result_dict["rate"]
        print(f"Reaction rate = {rate:+7.3e} [1/s]")
    atoms_list = []
    for datadict in data_dicts:
        atoms = reaction_mechanism.clean_surface.copy()
        atoms.symbols = datadict["elements"]
        atoms_list.append(atoms)
    write("bulks.traj", atoms_list)
    print("EVALUATING RATE FOR EQUIVALENT CLUSTER-SURFACES")
    reaction_mechanism = setup_reaction_mechanism(
        miller_index=miller_index,
        surface_type="cluster",
        calculator=calculator,
        features_bulk=features_bulk,
        features_gas=features_gas
    )

    original_indices = reaction_mechanism.clean_surface.info["indices_original"]
    print(original_indices)
    cluster_data_dicts = []
    for data_dict in data_dicts:
        print(len(data_dict["elements"]))
        symbols = [data_dict["elements"][index] for index in original_indices]
        result_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
            symbols=symbols
        )
        cluster_dict = {"elements":symbols, "batch":1, "score_dict":result_dict}
        cluster_data_dicts.append(cluster_dict)
        print(f"Symbols =", ",".join(symbols))
        rate = result_dict["rate"]
        print(f"Reaction rate = {rate:+7.3e} [1/s]")
    
    atoms_list = []
    for datadict in cluster_data_dicts:
        atoms = reaction_mechanism.clean_surface.copy()
        atoms.symbols = datadict["elements"]
        atoms_list.append(atoms)
    write("clusters.traj", atoms_list)

def setup_reaction_mechanism(
        miller_index:str,
        surface_type:str,
        calculator,
        features_bulk,
        features_gas
    ) -> ReactionMechanism:
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"../databases/{surface_type}_templates/{miller_index}"
    )
   
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        template_atoms_list=template_atoms_list,
        calculator=calculator,
        features_bulk=features_bulk,
        features_gas=features_gas,
        mechanism_pth_header="../yaml_files/reaction_mechanism"
    )
    return reaction_mechanism

def get_bulk_to_cluster_mapping(miller_index:str):
    if miller_index == "100":
        pass



# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
