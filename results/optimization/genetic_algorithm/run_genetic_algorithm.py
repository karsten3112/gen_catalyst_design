from gen_catalyst_design.utils import get_full_element_pool
from gen_catalyst_design.optimization import setup_optimization_objective
from gen_catalyst_design.optimization import Logger, evaluate_score_from_symbols
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from pygad import GA
import numpy as np

def main():
    template_type = "surface"
    miller_index = "100"
    element_pool = get_full_element_pool()
    num_samples = 10000
    n_candidates_per_generation = 100
    random_seed = 42
    include_stability = False
    do_post_processing = False

    
    reaction_mechanism, stabilizer, template_atoms_list = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header="../../../databases",
        yaml_files_header="../../../yaml_files",
        include_stability=include_stability,
    )

    search_kwargs = get_search_kwargs(
        sol_per_pop=n_candidates_per_generation
    )

    database = Database.establish_connection(
        filename="test_opt.db",
        database_kwargs={"append":False, "template_atoms_surf":template_atoms_list[0]}
    )
    
    datadicts = run_genetic_algorithm(
        num_samples=num_samples,
        element_pool=element_pool,
        reaction_mechanism=reaction_mechanism,
        stabilizer=stabilizer,
        database=database,
        random_seed=random_seed,
        search_kwargs=search_kwargs,
        objective_key="rate"
    )

    if do_post_processing:
        pass
        #Do something with datadicts maybe

    

def run_genetic_algorithm(
        num_samples:int,
        element_pool:list,
        reaction_mechanism:ReactionMechanism,
        database:Database=None,
        stabilizer:Stabilizer=None,
        random_seed:int=42,
        objective_key:str="rate",
        search_kwargs:dict={}
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
            objective_key=objective_key
        )
        #change the logging interval to 
        if "initial_population" not in search_kwargs and logger.n_obj_func_calls == search_kwargs["sol_per_pop"]:
            logger.log_interval-=search_kwargs["keep_elitism"]
        return score
    
    #print(f_call)

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


def get_search_kwargs(sol_per_pop:int=100):
    search_kwargs = {
        "sol_per_pop": sol_per_pop,
        "keep_elitism":1,
        "num_parents_mating": int(np.ceil(0.2*sol_per_pop)),
        "mutation_percent_genes": 10,
        "parent_selection_type": "tournament", # sss | rws | rank | random | tournament
        "crossover_type": "uniform",  # single_point | two_points | uniform
        "mutation_type": "random"  # random | swap | inversion | scramble
    }
    return search_kwargs
    


if __name__ == "__main__":
    main()