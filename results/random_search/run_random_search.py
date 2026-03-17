from gen_catalyst_design.utils import get_full_element_pool
from gen_catalyst_design.optimization import setup_optimization_objective
from gen_catalyst_design.optimization import Logger, evaluate_score_from_symbols
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from pygad import GA
import numpy as np
import random
import torch
import time

def main():
    template_type = "surface"
    miller_index = "100"
    element_pool = get_full_element_pool(["Mn", "Ga"])
    num_samples = 1000
    random_seed = 42
    include_stability = True
    
    reaction_mechanism, stabilizer, template_atoms_list = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header="../../databases",
        yaml_files_header="../../yaml_files",
        include_stability=include_stability,
    )

    database = Database.establish_connection(
        filename=f"result_{random_seed}_seed.db",
        #pth_header="rates_and_eform/large_element_pool",
        database_kwargs={"append":False, "template_atoms_surf":template_atoms_list[0]}
    )
    t_init = time.time()
    result = run_random_search(
        element_pool=element_pool,
        num_samples=num_samples,
        reaction_mechanism=reaction_mechanism,
        database=database,
        stabilizer=stabilizer,
        random_seed=random_seed
    )
    t_final = time.time()
    print(t_final-t_init)


def run_random_search(
        element_pool:list,
        num_samples:int,
        reaction_mechanism:ReactionMechanism,
        database:Database=None,
        stabilizer:Stabilizer=None,
        random_seed:int=42,
        objective_key:str="both",
        search_kwargs:dict={}
    ):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

    n_atoms_surf = len(reaction_mechanism.clean_surface)
    logger = Logger(
        database=database,
        log_interval=100
    )

    for _ in range(num_samples):
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        score = evaluate_score_from_symbols(
            symbols=symbols,
            reaction_mechanism=reaction_mechanism,
            logger=logger,
            stabilizer=stabilizer,
            objective_key=objective_key
        )
    logger.write_data_to_file()
    return logger.stored_datadicts

if __name__ == "__main__":
    main()