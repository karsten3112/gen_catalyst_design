from gen_catalyst_design.utils import get_features_bulk_and_gas, get_calculator, get_atoms_from_template_db
from gen_catalyst_design.optimization import Logger, evaluate_score_from_symbols
from gen_catalyst_design.db import Database
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from chgnet.model.dynamics import CHGNetCalculator
from scipy.optimize import dual_annealing
import numpy as np


def main():
    template_type = "cluster"
    miller_index = "100"
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    num_chains = 2
    num_samples = 100
    random_seed = 42
    include_stability = False

    features_bulk, features_gas = get_features_bulk_and_gas(pth_header="../../../yaml_files/features")
    
    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(
         model="WWL-GPR", 
         miller_index=miller_index
    )
    
    #Train calculator on database
    calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header="../../../databases/DFT_database",
         train_kwargs=train_kwargs
    )
    
    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"../../../databases/{template_type}_templates"
    )
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        template_atoms_list=template_atoms_list,
        calculator=calculator,
        features_bulk=features_bulk,
        features_gas=features_gas,
        mechanism_pth_header="../../../yaml_files/reaction_mechanism"
    )

    if include_stability and template_type == "surface":
        if template_type != "surface":
            stabilizer = None
            print(f"stability is set to be included, but cannot be estimated based on template type: {template_type}")
        else:
            stabilizer = Stabilizer(
                calculator=CHGNetCalculator(),
                template_atoms=template_atoms_list[0],
                ref_energy_file="chgnet_ref_energies.yaml",
                ref_energy_pth_header="../../../yaml_files/reference_energies",
                fmax=0.8
            )
    else:
        stabilizer = None


    search_kwargs = get_search_kwargs()

    database = Database.establish_connection(
        filename="test_opt.db",
        database_kwargs={"append":False, "template_atoms_surf":template_atoms_list[0]}
    )
    
    datadicts = run_dual_annealing(
        num_samples=num_samples,
        element_pool=element_pool,
        reaction_mechanism=reaction_mechanism,
        num_chains=num_chains,
        stabilizer=stabilizer,
        database=database,
        random_seed=random_seed,
        search_kwargs=search_kwargs,
        objective_key="rate"
    )

    #Do something with datadicts maybe

    
def run_dual_annealing(
        num_samples:int,
        element_pool:list,
        reaction_mechanism:ReactionMechanism,
        num_chains:int=10,
        database:Database=None,
        stabilizer:Stabilizer=None,
        random_seed:int=42,
        objective_key:str="rate",
        search_kwargs:dict={}
    ):

    n_atoms = len(reaction_mechanism.clean_surface)
    bounds = [(0, len(element_pool)-1)] * len(reaction_mechanism.clean_surface)
    
    samples_per_chain = int(np.ceil(num_samples/num_chains))
    maxiter = int(np.ceil((samples_per_chain-1)/2/n_atoms))

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
                 objective_key=objective_key
            )
            if logger.n_obj_func_calls == 1:
                logger.log_interval=2*n_atoms
            # Return the negative rate.
            return -score

    for ii in range(num_chains):
        result = dual_annealing(
            func=objective_func,
            bounds=bounds,
            maxfun=samples_per_chain,
            maxiter=maxiter,
            no_local_search=True,
            seed=random_seed+ii,
            **search_kwargs
        )
        logger.write_data_to_file()
        logger.reset_count_stats(log_interval=1)
    
    return logger.stored_datadicts


def get_search_kwargs():
    search_kwargs = {
         "visit":2.62,
         "initial_temp":5230.0,
         "restart_temp_ratio":0.001
    }
    return search_kwargs
    


if __name__ == "__main__":
    main()