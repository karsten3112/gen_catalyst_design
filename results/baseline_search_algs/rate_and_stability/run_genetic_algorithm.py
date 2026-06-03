from gen_catalyst_design.utils import get_full_element_pool, get_full_element_pool_no_saas
from gen_catalyst_design.optimization import setup_optimization_objective
from gen_catalyst_design.optimization import Logger, evaluate_score_from_symbols
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from distutils.util import strtobool
from ase.io import write
from pygad import GA
import numpy as np
import argparse
import os

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
    default=5,
)

parser.add_argument(
    "--elements",
    "-elems",
    type=str,
    required=False,
    default=None,
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
    "--rate_weight",
    "-r_w",
    type=float,
    required=False,
    default=0.5,
)


parser.add_argument(
    "--e_form_weight",
    "-e_form_w",
    type=float,
    required=False,
    default=0.5,
)

parser.add_argument(
    "--fmax",
    "-fmax",
    type=float,
    required=False,
    default=0.05,
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
    default="single_point",
)

parser.add_argument(
    "--parent_selection_type",
    "-selection_type",
    type=str,
    required=False,
    default="sss",
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
    "--setup_files_header",
    "-setup_files_header",
    type=str,
    required=False,
    default="../../gen_catalyst_design",
)

parser.add_argument(
    "--start_mean_lat",
    "-mean_lat",
    type=fbool,
    required=False,
    default=True,
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
    start_mean_lattice = parsed_args.start_mean_lat
    print(start_mean_lattice, type(start_mean_lattice))
    #Load full element_pool
    if parsed_args.elements is None:
        element_pool = get_full_element_pool_no_saas() #get_full_element_pool(["Ga", "Mn"])
    else:
        element_pool = parsed_args.elements.split(",")
    

    #Get surface type and whether stability is included
    template_type, include_stability = get_surface_params_from_target(
        target_type=objective_key
    )
    if include_stability:
        use_log = True
    
    print("======================RUNNNING GENETIC ALGORITHM======================")
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
        stability_kwargs={"fmax":parsed_args.fmax}
    )

    #Get search key-word arguments from input
    search_kwargs = get_search_kwargs(
        crossover_type=parsed_args.crossover_type,
        mutation_type=parsed_args.mutation_type,
        parent_selection_type=parsed_args.parent_selection_type
    )

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
    
    #Run the genetic algorithm
        datadicts, logged_atoms_conf = run_genetic_algorithm(
            num_samples=num_samples,
            element_pool=element_pool,
            reaction_mechanism=reaction_mechanism,
            stabilizer=stabilizer,
            database=database,
            random_seed=rnd_seed,
            search_kwargs=search_kwargs,
            objective_key=objective_key,
            score_weight_dict={"rate_weight":parsed_args.rate_weight, "e_form_weight":parsed_args.e_form_weight},
            start_from_mean_lattice=start_mean_lattice,
            use_log=use_log
        )

        database.close_connection()
        
        write_all_atoms_confs(
            logged_atoms_conf=logged_atoms_conf,
            filename=f"rnd_seed_{rnd_seed}_samples.traj",
            pth_header=parsed_args.outdir
        )


def write_all_atoms_confs(
        logged_atoms_conf:list,
        filename:str,
        pth_header:str=None
    ):
    if pth_header is not None:
        filename = os.path.join(pth_header, filename)
    all_confs = []
    for i, atoms_dict in enumerate(logged_atoms_conf):
        atoms_init, atoms_final = atoms_dict["atoms_init"], atoms_dict["atoms_final"]
        atoms_init.info["sample_num"] = i
        atoms_final.info["sample_num"] = i
        all_confs+=[atoms_init, atoms_final]
    write(filename=filename, images=all_confs)


def run_genetic_algorithm(
        num_samples:int,
        element_pool:list,
        reaction_mechanism:ReactionMechanism,
        database:Database=None,
        stabilizer:Stabilizer=None,
        random_seed:int=42,
        objective_key:str="rate",
        search_kwargs:dict={},
        score_weight_dict:dict={},
        start_from_mean_lattice:bool=True,
        use_log:bool=False
    ):

    logger = Logger(
        database=database,
        log_interval=search_kwargs["sol_per_pop"],
        match_log_interval_gen_iter=True
    )

    log_atoms_conf_list = []

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
            log_atoms_conf_list=log_atoms_conf_list,
            score_weight_dict=score_weight_dict,
            start_from_mean_lattice=start_from_mean_lattice,
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
    return logger.stored_datadicts, log_atoms_conf_list



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
        crossover_type:str="single_point",
        mutation_type:str="random",
        parent_selection_type:str="sss",
        mutation_percent_genes:float="default",
    ):
    search_kwargs = {
        "sol_per_pop": sol_per_pop,
        "keep_elitism":1,
        "num_parents_mating": int(np.ceil(0.2*sol_per_pop)),
        "mutation_percent_genes": mutation_percent_genes,
        "parent_selection_type": parent_selection_type, # sss | rws | rank | random | tournament
        "crossover_type": crossover_type,  # single_point | two_points | uniform
        "mutation_type": mutation_type  # random | swap | inversion | scramble
    }
    return search_kwargs
    


if __name__ == "__main__":
    main()