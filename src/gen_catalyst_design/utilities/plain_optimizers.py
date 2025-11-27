# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import time
import numpy as np
import random
from skopt import Optimizer as BayesOptimizer
from skopt.space import Categorical
from skopt.utils import use_named_args
from gen_catalyst_toolkit.db import Database
from skopt.callbacks import CheckpointSaver
from skopt import load

# -------------------------------------------------------------------------------------
# RANDOM SEARCH FUNCTION
# -------------------------------------------------------------------------------------

class Optimizer:
    def __init__(
            self, 
            reaction_rate_func: callable, 
            reaction_rate_kwargs: dict={},
            batch_size:int=100,
            batches:int=10,
            random_seed:int=42,
            database:Database=None,
            print_result:bool=False
        ):
        
        self.batch_size = batch_size
        self.batches = batches
        self.database = database
        self.random_seed = random_seed
        self.reaction_rate_func = reaction_rate_func
        self.reaction_rate_kwargs = reaction_rate_kwargs
        self.print_result = print_result
        self.timing_stat_dict = {"batch_size":self.batch_size,
                                 "batches":self.batches}

    def print_results(self, data_all:list):
        data_best = sorted(data_all, key=lambda xx: xx["rate"], reverse=True)[0]
        rate_best, symbols_best = data_best["rate"], data_best["symbols"]
        print(f"Best Structure of entire run with random seed {self.random_seed}:")
        print(f"Symbols =", ",".join(symbols_best))
        print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")

    def run_optimization(self, search_kwargs:dict={}) -> list:
        raise Exception("Must be implemented by sub-class")


class RandomSearch(Optimizer):
    def __init__(
            self, 
            reaction_rate_func, 
            reaction_rate_kwargs = {}, 
            batch_size = 100, 
            batches = 10, 
            random_seed = 42, 
            database = None, 
            print_result = False
        ):
        super().__init__(reaction_rate_func, reaction_rate_kwargs, batch_size, batches, random_seed, database, print_result)

    def run_optimization(self, element_pool:list, n_atoms_surf:int, search_kwargs = {}) -> list:
        random.seed(self.random_seed)
        data_all = []
        logging_times = []
        eval_times = []
        for batch in range(self.batches):
            data_batch = []
            eval_batch_init = time.time()
            for i in range(self.batch_size):
                symbols = random.choices(population=element_pool, k=n_atoms_surf)
                score_dict = self.reaction_rate_func(symbols=symbols, **self.reaction_rate_kwargs)
                data_batch.append({"elements":symbols, "batch": batch, "score_dict":score_dict})
            eval_batch_end = time.time()
            
            if self.database is not None:
                self.database.write_data_to_tables(data_dicts=data_batch, append=True)
            log_batch_end = time.time()

            eval_times.append(eval_batch_end-eval_batch_init)
            logging_times.append(log_batch_end-eval_batch_end)

            data_all+=data_batch

        if self.print_results is True:
            self.print_results(data_all=data_all)
        
        self.timing_stat_dict.update({"eval_times_per_batch":eval_times, "Logging_times_per_batch":logging_times})

        return data_all


# -------------------------------------------------------------------------------------
# BAYESIAN OPTIMIZATION USING GAUSSIAN PROCESSES
# -------------------------------------------------------------------------------------


class BayesianOptimizer(Optimizer): #need to test this method #Not sure if it works #Should produce exactly the same results as the method below
    def __init__(
            self, 
            reaction_rate_func, 
            reaction_rate_kwargs = {}, 
            batch_size = 100, 
            batches = 10, 
            random_seed = 42, 
            database = None, 
            print_result = False
        ):
        super().__init__(reaction_rate_func, reaction_rate_kwargs, batch_size, batches, random_seed, database, print_result)


    def run_optimization(self, element_pool: list, n_atoms_surf: int, search_kwargs = {}):
        batch_count = 0
        data_batch = []
        data_all = []

        space = [Categorical(element_pool, name=f"el_{ii}") for ii in range(n_atoms_surf)]
        
        @use_named_args(space)
        def objective_func(**kwargs):
            # Extract symbol list from kwargs.
            symbols = [kwargs[f"el_{ii}"] for ii in range(n_atoms_surf)]
            # Calculate reaction rate of the rate-determining step.
            score_dict = self.reaction_rate_func(symbols=symbols, **self.reaction_rate_kwargs)
            data_batch.append({"elements": symbols, "batch": batch_count, "score_dict":score_dict})
            rate = score_dict["rate"]
            # Return negative rate.
            return -rate
        

        eval_times = []
        logging_times = []

        optimizer = BayesOptimizer(
            dimensions=space,
            base_estimator="GP",
            random_state=self.random_seed,
            **search_kwargs
        )
        
        if "n_initial_points" in search_kwargs:
            n_init_points = search_kwargs["n_initial_points"]
            for _ in range(n_init_points):
                x = optimizer.ask()
                optimizer.tell(x=x, y=objective_func(x))
        else:
            raise Exception(f"No n_initial_points were provided in search_kwargs: {search_kwargs}")
    
        if n_init_points > self.batch_size:
            raise Exception(f"amount of initial samples cannot be larger than the batch size for now")
    
        for batch in range(self.batches):
            for _ in range(self.batch_size - n_init_points if batch == 0 else self.batch_size):
                x = optimizer.ask()
                optimizer.tell(x=x, y=objective_func(x))
            if self.database is not None:
                self.database.write_data_to_tables(data_dicts=data_batch, append=True)
            data_all+=data_batch
            data_batch=[]
            batch_count+=1
                
        self.timing_stat_dict.update({"eval_times_per_batch":eval_times, "Logging_times_per_batch":logging_times})
        return data_all


# -------------------------------------------------------------------------------------
# GET OPTIMIZER METHOD
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
# END
# -------------------------------------------------------------------------------------