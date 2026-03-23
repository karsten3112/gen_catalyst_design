from gen_catalyst_design.utils import get_full_element_pool
from gen_catalyst_design.optimization import setup_optimization_objective
from gen_catalyst_design.optimization import Logger, evaluate_score_from_symbols
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from distutils.util import strtobool
from pygad import GA
import numpy as np
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
    "--mutation_type",
    "-mut_type",
    type=str,
    required=False,
    default="random",
)

parser.add_argument(
    "--crossover_type",
    "-cross_type",
    type=str,
    required=False,
    default="uniform",
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
    default="genetic_alg.db",
)

parser.add_argument(
    "--outdir",
    "-dir",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "--setup_file_header",
    "-setup_file_header",
    type=str,
    required=False,
    default=None,
)


parsed_args = parser.parse_args()


def main():
    #Hyperparameter setup
    num_samples = 10000
    n_candidates_per_generation = 100
    miller_index = parsed_args.miller_index
    random_seed = parsed_args.random_seed
    objective_key = parsed_args.target
    use_log = parsed_args.use_log

    #Load full element_pool
    element_pool = get_full_element_pool()


    #Get surface type and whether stability is included
    template_type, include_stability = get_surface_params_from_target(
        target_type=objective_key
    )
    
    #Setup the reaction-mechanism -> Calculates the rate
    #Setup the stabilizer -> Estimates E_form
    #Get the template atoms used in both calculations
    reaction_mechanism, stabilizer, template_atoms_list = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header=f"{parsed_args.setup_files_header}/databases",
        yaml_files_header=f"{parsed_args.setup_files_header}/yaml_files",
        include_stability=include_stability,
    )

    #Get search key-word arguments from input
    search_kwargs = get_search_kwargs(
        sol_per_pop=n_candidates_per_generation,
        crossover_type=parsed_args.crossover_type,
        mutation_type=parsed_args.mutation_type
    )

    #Setup the database for storing the data
    database = Database.establish_connection(
        filename=parsed_args.db_filename,
        pth_header=parsed_args.outdir,
        database_kwargs={
            "append":False, 
            "template_atoms_surf":template_atoms_list[0]
        }
    )
    
    #Run the genetic algorithm
    datadicts = run_genetic_algorithm(
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




def run_genetic_algorithm(
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

    logger = Logger(
        database=database,
        log_interval=search_kwargs["sol_per_pop"],
        match_log_interval_gen_iter=True
    )

    n_elements = len(element_pool)
    num_generations = int(np.ceil(
        (num_samples - search_kwargs["sol_per_pop"]) / 
        (search_kwargs["sol_per_pop"] - search_kwargs["keep_elitism"])
    ))

    index_to_element = {ii: el for ii, el in enumerate(element_pool)}
  
    def fitness_func(ga_instance, solution, solution_idx):
        # Convert indices to element symbols.
        symbols = [index_to_element[int(ii)] for ii in solution]
        # Calculate reaction rate of the rate-determining step.
        score = evaluate_score_from_symbols(
            symbols=symbols,
            reaction_mechanism=reaction_mechanism,
            logger=logger,
            stabilizer=stabilizer,
            add_time_stats=True,
            objective_key=objective_key,
            use_log=use_log
        )
        #change the logging interval to 
        if "initial_population" not in search_kwargs and logger.n_obj_func_calls == search_kwargs["sol_per_pop"]:
            logger.log_interval-=search_kwargs["keep_elitism"]
        return score

    ga_instance = GA(
        num_generations=num_generations,
        fitness_func=fitness_func,
        num_genes=reaction_mechanism.n_atoms_surf,
        gene_type=int,
        init_range_low=0,
        init_range_high=n_elements,
        gene_space=list(range(n_elements)),
        random_mutation_min_val=0,
        random_mutation_max_val=n_elements-1,
        random_seed=random_seed,
        **search_kwargs,
    )

    ga_instance.run()
    #Write residual data if not written during last part searching
    logger.write_data_to_file()
    return logger.stored_datadicts



def get_surface_params_from_target(target_type:str="rate"):
    if target_type == "rate":
        return "cluster", False
    elif target_type == "stability":
        return "surface", True
    elif target_type == "both":
        return "surface", True
    else:
        raise Exception(f"target of type {target_type} is not implemented")


def get_search_kwargs(
        sol_per_pop:int=100, 
        crossover_type:str="uniform",
        mutation_type:str="random",
    ):
    search_kwargs = {
        "sol_per_pop": sol_per_pop,
        "keep_elitism":1,
        "num_parents_mating": int(np.ceil(0.2*sol_per_pop)),
        "mutation_percent_genes": 10,
        "parent_selection_type": "tournament", # sss | rws | rank | random | tournament
        "crossover_type": crossover_type,  # single_point | two_points | uniform
        "mutation_type": mutation_type  # random | swap | inversion | scramble
    }
    return search_kwargs
    


if __name__ == "__main__":
    main()