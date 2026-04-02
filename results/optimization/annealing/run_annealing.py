from gen_catalyst_design.utils import get_full_element_pool
from gen_catalyst_design.optimization import (
    setup_optimization_objective, Logger, evaluate_score_from_symbols, get_surface_params_from_target
)
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from distutils.util import strtobool
from scipy.optimize import dual_annealing
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))


parser.add_argument(
    "--random_seeds",
    "-rnd_seeds",
    type=str,
    required=False,
    default="42",
)

parser.add_argument(
    "--num_samples",
    "-n_samples",
    type=int,
    required=False,
    default=100,
)

parser.add_argument(
    "--target",
    "-t",
    type=str,
    required=False,
    default="rate",
)

parser.add_argument(
    "--miller_index",
    "-m_index",
    type=str,
    required=False,
    default="100",
)

parser.add_argument(
    "--use_log",
    "-use_log",
    type=fbool,
    required=False,
    default=False,
)

parser.add_argument(
    "--db_filename",
    "-filename",
    type=str,
    required=False,
    default="annealing.db",
)

parser.add_argument(
    "--outdir",
    "-dir",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "--setup_files_header",
    "-setup_files_header",
    type=str,
    required=False,
    default="../../gen_catalyst_design",
)

parser.add_argument(
    "--visit",
    "-visit",
    type=float,
    required=False,
    default=2.62,
)

parser.add_argument(
    "--init_temp",
    "-init_t",
    type=float,
    required=False,
    default=5230.0,
)

parser.add_argument(
    "--accept",
    "-acc",
    type=float,
    required=False,
    default=-5.0,
)

parser.add_argument(
    "--restart_temp_ratio",
    "-rs_t_rat",
    type=float,
    required=False,
    default=2e-5,
)

parsed_args = parser.parse_args()


def main():
    #Hyperparameter setup
    num_samples = parsed_args.num_samples
    miller_index = parsed_args.miller_index
    random_seeds = [int(rnd_seed) for rnd_seed in parsed_args.random_seeds.split(",")]
    objective_key = parsed_args.target
    use_log = parsed_args.use_log
    setup_files_header = parsed_args.setup_files_header
    #Load full element_pool
    element_pool = get_full_element_pool()
    

    #Get surface type and whether stability is included
    template_type, include_stability = get_surface_params_from_target(
        target_type=objective_key
    )
    
    print("======================RUNNNING ANNEALING ALGORITHM======================")
    print(f"Element pool chosen:")
    print(element_pool)
    print(f"facet: fcc-{miller_index}, template-type: {template_type}")
    print(f"objective target: {objective_key}, is log(rate) used: {use_log}")
    print(f"Is stability included: {include_stability}")

    #Setup the reaction-mechanism -> Calculates the rate
    #Setup the stabilizer -> Estimates E_form
    #Get the template atoms used in both calculations
    print("-------------------SETTING UP: REACTION-MECHANISM & STABILIZER-------------------")
    reaction_mechanism, stabilizer, template_atoms_list = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header=f"{setup_files_header}/databases" if setup_files_header is not None else None,
        yaml_files_header=f"{setup_files_header}/yaml_files" if setup_files_header is not None else None,
        include_stability=include_stability,
    )

    #Get search key-word arguments from input
    search_kwargs = {
        "visit":parsed_args.visit,
        "initial_temp":parsed_args.init_temp,
        "restart_temp_ratio":parsed_args.restart_temp_ratio,
        "accept":parsed_args.accept
    }

    print("-------------------SEARCH PARAMETERS-------------------")
    print(f"total amount of samples: {num_samples}")
    print(f"search key-word arguments set")
    print(search_kwargs)

    #Setup the database for storing the data
    for rnd_seed in random_seeds:
        database = Database.establish_connection(
            filename=f"rnd_seed_{rnd_seed}_samples.db",
            pth_header=parsed_args.outdir,
            database_kwargs={
                "append":False, 
                "template_atoms_surf":template_atoms_list[0]
            }
        )
    
        #Run the annealing
        datadicts = run_annealing(
            num_samples=num_samples,
            element_pool=element_pool,
            reaction_mechanism=reaction_mechanism,
            stabilizer=stabilizer,
            database=database,
            random_seed=rnd_seed,
            search_kwargs=search_kwargs,
            objective_key=objective_key,
            use_log=use_log
        )

        database.close_connection()


def run_annealing(
        num_samples:int,
        element_pool:list,
        reaction_mechanism:ReactionMechanism,
        database:Database=None,
        stabilizer:Stabilizer=None,
        random_seed:int=42,
        objective_key:str="rate",
        search_kwargs:dict={},
        use_log:bool=False
    ):

    n_atoms = len(reaction_mechanism.clean_surface)
    bounds = [(0, len(element_pool)-1)] * len(reaction_mechanism.clean_surface)

    logger = Logger(
        database=database,
        log_interval=1,
        match_log_interval_gen_iter=True
    )

    def objective_func(xx):
            # xx is an array of floats, map to nearest integer.
            x_int = [int(round(ii)) for ii in xx]
            symbols = [element_pool[ii] for ii in x_int]
            # Calculate reaction rate of the rate-determining step.
            score = evaluate_score_from_symbols(
                 symbols=symbols,
                 reaction_mechanism=reaction_mechanism,
                 logger=logger,
                 stabilizer=stabilizer,
                 objective_key=objective_key,
                 use_log=use_log
            )
            if logger.n_obj_func_calls == 1:
                logger.log_interval=2*n_atoms
            # Return the negative rate.
            return -score

    result = dual_annealing(
        func=objective_func,
        bounds=bounds,
        maxfun=num_samples,
        maxiter=num_samples,
        no_local_search=True,
        seed=random_seed,
        **search_kwargs
    )
    logger.write_data_to_file()
    logger.reset_count_stats(log_interval=1)
    return logger.stored_datadicts


if __name__ == "__main__":
    main()