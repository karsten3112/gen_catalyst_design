# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------
from distutils.util import strtobool
import numpy as np
import argparse
import os
import sys
import time
import torch
import random
import yaml
sys.path.insert(0, "../../../")
from gen_catalyst_toolkit.utilities.databases import connect, write_data_to_db, load_data_from_db, clear_database
from gen_catalyst_toolkit.reaction_rate_calculation import get_calculator, get_features_bulk_and_gas, get_atoms_from_template_db, reaction_rate_of_RDS_from_symbols
from gen_catalyst_toolkit.utilities.reaction_rates import ReactionMechanism
#from gen_catalyst_toolkit.utilities.calculators import Calculator

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
    default=10,
)

parser.add_argument(
    "--miller_index",
    "-m_index",
    type=str,
    required=True
)

parser.add_argument(
    "--n_samples_per_batch",
    "-n_batch_samples",
    type=int,
    required=False,
    default=1000,
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
    "--show_results",
    "-print_res",
    type=bool,
    required=False,
    default=False,
)

parser.add_argument(
    "--write_results",
    "-write_res",
    type=bool,
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

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    start_time = time.time()
    get_timings = True
    miller_index = parsed_args.miller_index # 100 | 111 | 211
    random_seed = parsed_args.random_seed   # Random seed for reproducibility.
    n_samples_per_batch = parsed_args.n_samples_per_batch # Number of structures evaluated per run.
    n_batches = parsed_args.n_batches # Number of search runs.
    print_results = parsed_args.show_results
    write_results = parsed_args.write_results
    model = parsed_args.model

    element_pool = ["Rh", "Cu", "Au"] #Maybe give this as input

    universal_pth_header = "/home/kwalz/speciale_files/gen_catalyst_toolkit"
    results_dir = parsed_args.out_dir
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

    # Get features.    
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header=os.path.join(universal_pth_header, "yaml_files/features"))

    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    train_init_time = time.time()
    calculator.train_model_from_db(
        db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
        features_bulk=features_bulk, 
        features_gas=features_gas, 
        db_pth_header=os.path.join(universal_pth_header, "databases/DFT_database"),
        train_kwargs=train_kwargs
    )
    train_end_time = time.time()

    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db", 
        pth_header=os.path.join(universal_pth_header, f"databases/templates/{miller_index}")
    )

    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        calculator=calculator, 
        mechanism_pth_header=os.path.join(universal_pth_header, "yaml_files/reaction_mechanism")
    )

    reaction_rate_kwargs = {"reaction_mechanism":reaction_mechanism,
                            "template_atoms_list":template_atoms_list,
                            "features_bulk":features_bulk,
                            "features_gas":features_gas,
                            "n_atoms_surf":n_atoms_surf
    }

    eval_begin_time = time.time()
    # Run multiple searches.
    eval_time_diffs = []
    log_time_diffs = []
    for run_id in range(n_batches):
        eval_init_time = time.time()
        data_run = run_random_search(
            reaction_rate_fun=reaction_rate_of_RDS_from_symbols, 
            reaction_rate_kwargs=reaction_rate_kwargs,
            element_pool=element_pool,
            n_atoms_surf=n_atoms_surf,
            n_eval=n_samples_per_batch,
            run_id=run_id,
            #random_seed=random_seed,
            print_results=print_results,
            )
        eval_end_time = time.time()
        eval_time_diffs.append(eval_end_time - eval_init_time)

        if write_results:
            data_db = connect(filename=f"random_seed_{random_seed}_results.db", pth_header=results_dir)
            write_data_to_db(database=data_db, data_list=data_run, append=True)
        log_end_time = time.time()
        log_time_diffs.append(log_end_time-eval_end_time)
    
    end_time = time.time()

    if get_timings is True:
        timing_dict = {"total_time": end_time-start_time, 
                       "training_time":train_end_time-train_init_time,
                       "param_load":train_init_time - start_time,
                       "template_db_load":eval_begin_time-train_end_time,
                       "batch_size":n_samples_per_batch,
                       "batches": n_batches,
                       "eval_times":float(np.sum(eval_time_diffs)),
                       "logging_times":float(np.sum(log_time_diffs)),
                       "eval_time_per_batch": float(np.mean(eval_time_diffs)),
                       "log_time_per_batch": float(np.mean(log_time_diffs))
                       }
        with open(os.path.join(results_dir,'timing_data.yaml'), 'w') as fileobj:
            yaml.safe_dump(timing_dict, fileobj, default_flow_style=False)

# -------------------------------------------------------------------------------------
# RUN RANDOM SEARCH
# -------------------------------------------------------------------------------------

def run_random_search(
    reaction_rate_fun: callable,
    reaction_rate_kwargs: dict,
    element_pool: list,
    n_atoms_surf: int,
    n_eval: int,
    run_id: int,
    #random_seed: int,
    print_results: bool = True,
    #search_kwargs: dict = {},
    data_input: list = None
):
    """
    Run a structure optimization with the random search method.
    """
    # Prepare data storage for the run.
    data_run = data_input or []
    # Random search of surface with highest reaction rate.
    #random.seed(random_seed)#+run_id) #Not sure about the random seed here
    for jj in range(n_eval):
        # Get elements for the surface.
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        # Calculate reaction rate.
        score_dict = reaction_rate_fun(symbols=symbols, **reaction_rate_kwargs)
        data_run.append({"elements": symbols, "run_ID": run_id, "scores":score_dict})
        # Print results to screen.
        if print_results is True:
            rate = score_dict["TOF"]
            print(f"Symbols =", ",".join(symbols))
            print(f"Reaction Rate = {rate:+7.3e} [1/s]")
    # Get best structure.
    if print_results is True:
        data_best = sorted(data_run, key=lambda xx: xx["rate"], reverse=True)[0]
        rate_best, symbols_best = data_best["rate"], data_best["symbols"]
        print(f"Best Structure of run {run_id}:")
        print(f"Symbols =", ",".join(symbols_best))
        print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")
    # Return run data.
    return data_run

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------