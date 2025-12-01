# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------
from distutils.util import strtobool
import numpy as np
import argparse
import os
import yaml
from catalyst_opt_tools.utilities import update_atoms_list
from gen_catalyst_design.db import Database
from reaction_rate_calculation import (
    get_calculator, 
    get_features_bulk_and_gas, 
    get_atoms_from_template_db, 
    reaction_rate_of_RDS_from_symbols
)
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.optimization import get_optimization_method, Optimizer

# -------------------------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--n_samples",
    "-n",
    type=int,
    required=False,
    default=2,
)

parser.add_argument(
    "--log_interval",
    "-log_int",
    type=int,
    required=False,
    default=2,
)

parser.add_argument(
    "--miller_index",
    "-m_index",
    type=str,
    required=True
)

parser.add_argument(
    "--random_seed",
    "-rnd_seed",
    type=int,
    required=True
)

parser.add_argument(
    "--out_dir",
    "-out",
    type=str,
    required=False,
    default="results",
)

parser.add_argument(
    "--print_results",
    "-print_res",
    type=fbool,
    required=False,
    default=False,
)

parser.add_argument(
    "--write_results",
    "-w_res",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--model",
    "-model",
    type=str,
    required=False,
    default="WWL-GPR",
)

parser.add_argument(
    "--opt_method",
    "-opt",
    type=str,
    required=False,
    default="RandomSearching",
)

parser.add_argument(
    "--element_pool",
    "-elem_pool",
    type=str,
    required=False,
    default="Rh,Cu,Au",
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    global_pth_header = "../" #/home/kwalz/speciale_files/gen_catalyst_toolkit"
    miller_index = parsed_args.miller_index # 100 | 111 | 211
    random_seed = parsed_args.random_seed   # Random seed for reproducibility.
    results_dir = parsed_args.out_dir

    print_results = parsed_args.print_results
    write_results = parsed_args.write_results

    model = parsed_args.model

    #Get element-pool from input
    element_pool = parsed_args.element_pool.split(",")
    
    #Get features.    
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header=os.path.join(global_pth_header, "yaml_files/features"))

    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    calculator.train_model_from_db(
        db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
        features_bulk=features_bulk, 
        features_gas=features_gas, 
        db_pth_header=os.path.join(global_pth_header, "databases/DFT_database"),
        train_kwargs=train_kwargs
    )

    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db", 
        pth_header=os.path.join(global_pth_header, f"databases/templates/{miller_index}")
    )

    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        calculator=calculator, 
        mechanism_pth_header=os.path.join(global_pth_header, "yaml_files/reaction_mechanism")
    )

    #Get reaction rate arguments
    reaction_rate_kwargs = {
        "reaction_mechanism":reaction_mechanism,
        "template_atoms_list":template_atoms_list,
        "features_bulk":features_bulk,
        "features_gas":features_gas,
        "n_atoms_surf":n_atoms_surf
    }

    optimizer, search_kwargs = get_optimization_method(
        template_atoms_surf=template_atoms_list[0],
        method=parsed_args.opt_method,
        element_pool=element_pool,
        reaction_rate_func=reaction_rate_of_RDS_from_symbols,
        filename=f"runID_{random_seed}_results.db",
        pth_header=results_dir,
        random_seed=random_seed,
        log_interval=parsed_args.log_interval,
        reaction_rate_kwargs=reaction_rate_kwargs
    )

    if parsed_args.opt_method == "ScikitOptimizer":
        search_kwargs.update({"n_initial_points": int(parsed_args.n_samples / 10)})

    results = optimizer.run_optimization(
        n_samples=parsed_args.n_samples, 
        search_kwargs = search_kwargs, 
        print_result=print_results, 
        print_progress=print_results
    )



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


if __name__ == "__main__":
    main()