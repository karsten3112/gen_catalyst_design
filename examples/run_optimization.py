# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------
from distutils.util import strtobool
import numpy as np
import argparse
import os
import time
import torch
import random
import yaml
from gen_catalyst_design.db import Database
from .reaction_rate_calculation import (
    get_calculator, 
    get_features_bulk_and_gas, 
    get_atoms_from_template_db, 
    reaction_rate_of_RDS_from_symbols
)
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.utilities.plain_optimizers import RandomSearch, BayesianOptimizer

# -------------------------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--n_batches",
    "-n_batches",
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
    "--batch_size",
    "-batch_size",
    type=int,
    required=False,
    default=5,
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
    default="random_search",
)

parser.add_argument(
    "--timings",
    "-time",
    type=fbool,
    required=False,
    default=True,
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
    start_time = time.time()
    global_pth_header = "../gen_catalyst_toolkit" #/home/kwalz/speciale_files/gen_catalyst_toolkit"
    get_timings = parsed_args.timings
    miller_index = parsed_args.miller_index # 100 | 111 | 211
    random_seed = parsed_args.random_seed   # Random seed for reproducibility.
    batch_size = parsed_args.batch_size # Number of structures evaluated per batch.
    n_batches = parsed_args.n_batches # Number of batches per run.
    results_dir = parsed_args.out_dir

    print_results = parsed_args.print_results
    write_results = parsed_args.write_results

    
    model = parsed_args.model

    #Get element-pool from input
    element_pool = [elem for elem in parsed_args.element_pool.split(",")]
    

    # Get features.    
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header=os.path.join(global_pth_header, "yaml_files/features"))

    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    train_init_time = time.time()
    calculator.train_model_from_db(
        db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
        features_bulk=features_bulk, 
        features_gas=features_gas, 
        db_pth_header=os.path.join(global_pth_header, "databases/DFT_database"),
        train_kwargs=train_kwargs
    )
    train_end_time = time.time()

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
    reaction_rate_kwargs = {"reaction_mechanism":reaction_mechanism,
                            "template_atoms_list":template_atoms_list,
                            "features_bulk":features_bulk,
                            "features_gas":features_gas,
                            "n_atoms_surf":n_atoms_surf
    }

    #Initialize database for logging
    if write_results is True:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        database = Database.establish_connection(
            filename=f"runID_{random_seed}_results.db", 
            miller_index=miller_index,
            use_tempdir=True,
            pth_header=results_dir
        )
    else:
        database = None

    #Drop existing tables, so no duplicates are present
    drop_init = time.time()
    database.drop_tables_db_file()
    drop_time = time.time() - drop_init


    #Initialize optimizer
    optimizer, search_kwargs = get_optimizer(
        method=parsed_args.opt_method,
        reaction_rate_func=reaction_rate_of_RDS_from_symbols, 
        reaction_rate_kwargs=reaction_rate_kwargs, 
        batch_size=batch_size,
        batches=n_batches,
        random_seed=random_seed,
        database=database,
        print_result=print_results
    )

    #Run optimization
    eval_begin_time = time.time()
    results = optimizer.run_optimization(
        element_pool=element_pool, 
        n_atoms_surf=n_atoms_surf, 
        search_kwargs=search_kwargs
    )
    end_time = time.time()

    if get_timings is True:
        timing_stat_dict = optimizer.timing_stat_dict
        timing_stat_dict.update(
            {"total_time": end_time-start_time, 
             "training_time":train_end_time-train_init_time,
             "param_load":train_init_time - start_time,
             "template_db_load":eval_begin_time-train_end_time,
             "drop_time":drop_time
        })
        with open(os.path.join(results_dir,f'runID_{random_seed}_timing_data.yaml'), 'w') as fileobj:
            yaml.safe_dump(timing_stat_dict, fileobj, default_flow_style=False)
    
    database.close_connection()


# -------------------------------------------------------------------------------------
# GET OPTIMIZER CLASS AND SEARCH KWARGS
# -------------------------------------------------------------------------------------

def get_optimizer(
        method:str, 
        reaction_rate_func:callable, 
        reaction_rate_kwargs:dict, 
        batch_size:int=100, 
        batches:int=10, 
        random_seed:int=42, 
        database:Database=None, 
        print_result:bool=False
    ):
    if method == "random_search":
        optimizer = RandomSearch(
            reaction_rate_func=reaction_rate_func, 
            reaction_rate_kwargs=reaction_rate_kwargs, 
            batch_size=batch_size, 
            batches=batches,
            random_seed=random_seed,
            database=database,
            print_result=print_result
        )
        search_kwargs = {}
    elif method == "bayesian_opt":
        optimizer = BayesianOptimizer(
            reaction_rate_func=reaction_rate_func, 
            reaction_rate_kwargs=reaction_rate_kwargs, 
            batch_size=batch_size, 
            batches=batches,
            random_seed=random_seed,
            database=database,
            print_result=print_result
        )
        search_kwargs = {"n_initial_points": 10, "acq_func": "EI"}
    else:
        raise Exception(f"Optimizer of type {method} has not been implemented")
    return optimizer, search_kwargs


# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------