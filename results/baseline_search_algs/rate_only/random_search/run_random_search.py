from gen_catalyst_design.utils import get_full_element_pool
from gen_catalyst_design.optimization import (
    setup_optimization_objective, Logger, evaluate_score_from_symbols, get_surface_params_from_target
)
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from distutils.util import strtobool
import random
import argparse

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))


parser.add_argument(
    "--random_seed",
    "-rnd_seed",
    type=int,
    required=False,
    default=42,
)

parser.add_argument(
    "--num_samples",
    "-n_samples",
    type=int,
    required=False,
    default=10000,
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
    default="random_search.db",
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
    default="../../..",
)


parsed_args = parser.parse_args()


def main():
    #Hyperparameter setup
    num_samples = parsed_args.num_samples
    miller_index = parsed_args.miller_index
    random_seed = parsed_args.random_seed
    objective_key = parsed_args.target
    use_log = parsed_args.use_log
    setup_files_header = parsed_args.setup_files_header
    #Load full element_pool
    element_pool = get_full_element_pool()
    

    #Get surface type and whether stability is included
    template_type, include_stability = get_surface_params_from_target(
        target_type=objective_key
    )
    
    print("======================RUNNNING RANDOM SEARCH ALGORITHM======================")
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
    search_kwargs = {}

    print("-------------------SEARCH PARAMETERS-------------------")
    print(f"total amount of samples: {num_samples}")
    print(f"search key-word arguments set")
    print(search_kwargs)

    #Setup the database for storing the data
    database = Database.establish_connection(
        filename=parsed_args.db_filename,
        pth_header=parsed_args.outdir,
        database_kwargs={
            "append":False, 
            "template_atoms_surf":template_atoms_list[0]
        }
    )
    
    #Run the random search
    datadicts = run_random_search(
        num_samples=num_samples,
        element_pool=element_pool,
        reaction_mechanism=reaction_mechanism,
        stabilizer=stabilizer,
        database=database,
        random_seed=random_seed,
        search_kwargs=search_kwargs,
        objective_key=objective_key,
        use_log=use_log
    )


def run_random_search(
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
    random.seed(random_seed)
    n_atoms_surf = len(reaction_mechanism.clean_surface)

    logger = Logger(
        database=database,
        log_interval=100,
        match_log_interval_gen_iter=True
    )

    for _ in range(num_samples):
        symbols = random.choices(population=element_pool, k=n_atoms_surf)

        score = evaluate_score_from_symbols(
            symbols=symbols,
            reaction_mechanism=reaction_mechanism,
            logger=logger,
            stabilizer=stabilizer,
            add_time_stats=True,
            objective_key=objective_key,
            use_log=use_log
        )
    logger.write_data_to_file()
    return logger.stored_datadicts




if __name__ == "__main__":
    main()