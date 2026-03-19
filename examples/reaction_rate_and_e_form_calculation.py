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
from gen_catalyst_design.stability import Stabilizer, apply_inversion_symmetry
from chgnet.model.dynamics import CHGNetCalculator
import torch
from ase.io import read, write


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    model = "WWL-GPR"
    surface_type = "surface" # cluster | surface
    miller_index = "100" # 100 | 111
    element_pool = ["Rh", "Cu", "Au"] # Elements of the surface.
    random_seed = 42 # Random seed for reproducibility.
    n_samples = 1
    atoms_traj_file = None
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    
    # Get features.
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header="../yaml_files/features")
    
    #Get calculator of model type and training parameters
    rate_calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    rate_calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header="../databases/DFT_database",
         train_kwargs=train_kwargs
    )
    
    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"../databases/{surface_type}_templates"
    )
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        template_atoms_list=template_atoms_list,
        calculator=rate_calculator,
        features_bulk=features_bulk,
        features_gas=features_gas,
        mechanism_pth_header="../yaml_files/reaction_mechanism"
    )

    stabilizer = Stabilizer(
        template_atoms=template_atoms_list[0],
        calculator=CHGNetCalculator(),
        ref_energy_file="chgnet_ref_energies.yaml",
        ref_energy_pth_header="../yaml_files/reference_energies",
        fmax=0.05
    )
    datadicts = []
    template_atoms = reaction_mechanism.clean_surface.copy()
    for i in range(n_samples):
        print(f"EVALUATION SURFACE NUM {i+1}")
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        template_atoms.symbols = symbols
        inverted_example = apply_inversion_symmetry(
            atoms=template_atoms.copy(),
            miller_index=miller_index
        )
        write("example_inverted.png", images=[inverted_example], rotation='10z,-60x')
        exit()
        result_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(symbols=symbols)
        rate = result_dict["rate"]
        e_form_dict = stabilizer.get_formation_energy_from_symbols(symbols=symbols, trajectory="test_relax.traj")
        e_form = e_form_dict["e_form"]
        result_dict.update({"e_form":e_form})
        data_dict = {"elements":symbols, "batch":1, "score_dict":result_dict}
        datadicts.append(data_dict)
        print("---------------------------------------------")
        print(f"elements = {symbols}")
        print(f"Reaction rate = {rate:+7.3e} [1/s]")
        print(f"Formation energy = {e_form:+7.3e} [eV]")
        print("---------------------------------------------")

    database = Database.establish_connection(
        filename="test_pred.db",
        database_kwargs={"append":False, "template_atoms_surf":reaction_mechanism.clean_surface} 
        #surface_type=surface_type, 
        #add_e_form=True,
        #miller_index=miller_index
    )
    database.write_data_to_tables(data_dicts=datadicts)

if __name__ == "__main__":
    main()
