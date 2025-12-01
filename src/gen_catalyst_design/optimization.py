# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------


import random
import os
import sys
import numpy as np
from .db import Database
from ase.atoms import Atoms
from catalyst_opt_tools.optimization import print_search_progress, print_search_results
from skopt.optimizer import base_minimize
from scipy.optimize import dual_annealing
from skopt.space import Categorical
from skopt.utils import use_named_args
from ase_ml_models.yaml import write_to_yaml
from pygad import GA

# -------------------------------------------------------------------------------------
# LOGGER CLASS
# -------------------------------------------------------------------------------------

class Logger:
    def __init__(
            self,
            miller_index:str,
            filename:str,
            pth_header:str=None,
            log_interval:int=100,
            database_kwargs:dict={}
        ):
        self.pth_header = pth_header
        if self.pth_header is not None and not os.path.exists(self.pth_header):
            os.makedirs(self.pth_header)
        
        self.database = None
        self.filetype = filename.split(".")[-1]
        if self.filetype == "db":
            self.database = Database.establish_connection(
                filename=filename,
                miller_index=miller_index,
                pth_header=pth_header,
                **database_kwargs
            )
        elif self.filetype == "yaml":
            pass
        else:
            raise Exception(f"Filetype of type: {self.filetype} is not implemented")
        
        self.filename = filename
        self.pth_header = pth_header
        self.log_interval = log_interval
        self.log_counter = 0
        self.logged_indices = 0
        self.times_logged = 0

    def write_data_to_file(self, data_dicts, append):
        if self.filetype == "db":
            self.database.write_data_to_tables(data_dicts=data_dicts, append=append)
        elif self.filetype == "yaml":
            if self.pth_header is not None:
                filename = os.path.join(self.pth_header, self.filename)
            else:
                filename = self.filename
            write_to_yaml(
                filename=filename
            )
        else:
            raise Exception("File of this type does not exist")
    
    def filter_data_for_logging(self, data_dicts:list):
        data_dicts_for_logging = data_dicts[self.logged_indices:]
        for data_dict in data_dicts_for_logging:
                data_dict.update({"batch":self.times_logged})
        return data_dicts_for_logging

    def log_data(self, data_dicts:list):
        if self.log_counter == self.log_interval-1:
            filtered_dicts = self.filter_data_for_logging(data_dicts=data_dicts)
            self.write_data_to_file(data_dicts=filtered_dicts, append=True)
            self.logged_indices += len(data_dicts) - self.logged_indices
            self.log_counter=0
            self.times_logged+=1
        else:
            self.log_counter+=1

    def log_residual_data(self, data_dicts:list):
        if len(data_dicts) > self.logged_indices:
            filtered_dicts = self.filter_data_for_logging(data_dicts=data_dicts)
            self.write_data_to_file(data_dicts=filtered_dicts, append=True)
            self.logged_indices += len(data_dicts) - self.logged_indices
            self.log_counter=0
            self.times_logged+=1


# -------------------------------------------------------------------------------------
# OPTIMIZER BASE CLASS
# -------------------------------------------------------------------------------------

class Optimizer:
    def __init__(
            self,
            filename:str,
            element_pool:list,
            template_atoms_surf:Atoms,
            reaction_rate_func: callable,
            pth_header:str=None,
            random_seed:int=42,
            log_interval:int=100,
            reaction_rate_kwargs:dict={},
            database_kwargs:dict={}
        ):
        random.seed(random_seed)
        self.element_pool = element_pool
        self.logger = Logger(
            miller_index=template_atoms_surf.info["miller_index"],
            filename=filename,
            pth_header=pth_header,
            log_interval=log_interval,
            database_kwargs=database_kwargs
        )
        self.reaction_rate_kwargs = reaction_rate_kwargs
        self.random_seed = random_seed
        self.reaction_rate_func = reaction_rate_func
        self.n_atoms_surf = len(template_atoms_surf)

    def print_results(self, data_all:list):
        data_best = sorted(data_all, key=lambda xx: xx["rate"], reverse=True)[0]
        rate_best, symbols_best = data_best["rate"], data_best["symbols"]
        print(f"Best Structure of entire run with random seed {self.random_seed}:")
        print(f"Symbols =", ",".join(symbols_best))
        print(f"Reaction Rate = {rate_best:+7.3e} [1/s]")

    def run_optimization(self, n_samples:int, search_kwargs:dict={}) -> list:
        raise Exception("Must be implemented by sub-class")

# -------------------------------------------------------------------------------------
# RANDOM SEARCHING
# -------------------------------------------------------------------------------------

class RandomSearcher(Optimizer):
    def __init__(self, filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header = None, random_seed = 42, log_interval = 100, reaction_rate_kwargs = {}, database_kwargs = {}):
        super().__init__(filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header, random_seed, log_interval, reaction_rate_kwargs, database_kwargs)

    def run_optimization(self, n_samples, search_kwargs = {}, print_result:bool=False, print_progress:bool=False):
        data_dicts = []
        for jj in range(n_samples):
            symbols = random.choices(population=self.element_pool, k=self.n_atoms_surf)
            result_dict = self.reaction_rate_func(symbols=symbols, **self.reaction_rate_kwargs)
            data_dict = {"elements":symbols, "score_dict":result_dict}
            data_dicts.append(data_dict)
            self.logger.log_data(data_dicts=data_dicts)
            if print_progress:
                print_search_progress(
                    run_id=self.random_seed,
                    nn=jj,
                    n_eval=n_samples
                )
        self.logger.log_residual_data(data_dicts=data_dicts)
        if print_result:
            data = sorted(data_dicts, key=lambda xx: xx["score_dict"]["rate"], reverse=True)[0]
            rate, symbols = data["score_dict"]["rate"], data["elements"]
            print(f"Best Structure of Run {self.random_seed}:")
            print_search_results(symbols=symbols, rate=rate)
        return data_dicts


# -------------------------------------------------------------------------------------
# SCIKIT OPTIMIZER
# -------------------------------------------------------------------------------------

class ScikitOptimizer(Optimizer):
    def __init__(self, filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header = None, random_seed = 42, log_interval = 100, reaction_rate_kwargs = {}, database_kwargs = {}):
        super().__init__(filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header, random_seed, log_interval, reaction_rate_kwargs, database_kwargs)
    
    def run_optimization(self, n_samples, search_kwargs = {}, print_result:bool=False, print_progress:bool=False):
        data_dicts = []
        space = [Categorical(self.element_pool, name=f"el_{ii}") for ii in range(self.n_atoms_surf)]
        @use_named_args(space)
        def objective_func(**kwargs):
            # Extract symbol list from kwargs.
            symbols = [kwargs[f"el_{ii}"] for ii in range(self.n_atoms_surf)]
            # Calculate reaction rate of the rate-determining step.
            result_dict = self.reaction_rate_func(symbols=symbols, **self.reaction_rate_kwargs)
            rate = result_dict["rate"]
            data_dict = {"elements":symbols, "score_dict":result_dict}
            data_dicts.append(data_dict)
            self.logger.log_data(data_dicts=data_dicts)
            # Return the negative rate.
            return -rate
        # Run the scikit optimization.
        result = base_minimize(
            func=objective_func,
            dimensions=space,
            n_calls=n_samples - 1,
            random_state=self.random_seed,
            **search_kwargs,
        )
        self.logger.log_residual_data(data_dicts=data_dicts)
        if print_result is True:
            data = sorted(data_dicts, key=lambda xx: xx["score_dict"]["rate"], reverse=True)[0]
            rate, symbols = data["rate"], data["elements"]
            print(f"Best Structure of Run {self.random_seed}:")
            print_search_results(symbols=symbols, rate=rate)
        return data_dicts

# -------------------------------------------------------------------------------------
# DUAL ANNEALING OPTIMIZER
# -------------------------------------------------------------------------------------

class DualAnnealing(Optimizer):
    def __init__(self, filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header = None, random_seed = 42, log_interval = 100, reaction_rate_kwargs = {}, database_kwargs = {}):
        super().__init__(filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header, random_seed, log_interval, reaction_rate_kwargs, database_kwargs)
    
    def run_optimization(self, n_samples, search_kwargs = {}, print_result:bool=False, print_progress:bool=False):
        data_dicts = []
        def objective_fun(xx):
            # xx is an array of floats, map to nearest integer.
            x_int = [int(round(ii)) for ii in xx]
            symbols = [self.element_pool[ii] for ii in x_int]
            # Calculate reaction rate of the rate-determining step.
            result_dict = self.reaction_rate_func(symbols=symbols, **self.reaction_rate_kwargs)
            rate = result_dict["rate"]
            if len(data_dicts) >= n_samples - 1:
                return -rate
            data_dict = {"elements":symbols, "score_dict":result_dict}
            data_dicts.append(data_dict)
            self.logger.log_data(data_dicts=data_dicts)
            # Return the negative rate.
            return -rate
        # Perform dual annealing optimization.
        bounds = [(0, len(self.element_pool)-1)] * self.n_atoms_surf
        result = dual_annealing(
            func=objective_fun,
            bounds=bounds,
            maxfun=n_samples - 1,
            seed=self.random_seed,
        )
        self.logger.log_residual_data(data_dicts=data_dicts)
        if print_result is True:
            data = sorted(data_dicts, key=lambda xx: xx["score_dict"]["rate"], reverse=True)[0]
            rate, symbols = data["score_dict"]["rate"], data["elements"]
            print(f"Best Structure of Run {self.random_seed}:")
            print_search_results(symbols=symbols, rate=rate)
        return data_dicts


# -------------------------------------------------------------------------------------
# GENETIC OPTIMIZER
# -------------------------------------------------------------------------------------

class GeneticAlgorithm(Optimizer):
    def __init__(self, filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header = None, random_seed = 42, log_interval = 100, reaction_rate_kwargs = {}, database_kwargs = {}):
        super().__init__(filename, element_pool, template_atoms_surf, reaction_rate_func, pth_header, random_seed, log_interval, reaction_rate_kwargs, database_kwargs)

    def run_optimization(self, n_samples, search_kwargs = {}, print_result:bool=False, print_progress:bool=False):
        data_dicts = []
        num_generations = int(np.ceil(
            (n_samples - search_kwargs["sol_per_pop"]) / 
            (search_kwargs["sol_per_pop"] - search_kwargs["keep_parents"])
        ))
        # Convert elements list to index and back.
        index_to_element = {ii: el for ii, el in enumerate(self.element_pool)}
        n_elements = len(self.element_pool)
        # Fitness function.
        def fitness_func(ga_instance, solution, solution_idx):
            # Convert indices to element symbols.
            symbols = [index_to_element[int(ii)] for ii in solution]
            # Calculate reaction rate of the rate-determining step.
            result_dict = self.reaction_rate_func(symbols=symbols, **self.reaction_rate_kwargs)
            rate = result_dict["rate"]
            if len(data_dicts) >= n_samples - 1:
                return rate
            data_dict = {"elements":symbols, "score_dict":result_dict}
            data_dicts.append(data_dict)
            self.logger.log_data(data_dicts=data_dicts)
            
            return rate
        # Set up the Genetic Algorithm.
        ga_instance = GA(
            num_generations=num_generations,
            fitness_func=fitness_func,
            num_genes=self.n_atoms_surf,
            gene_type=int,
            init_range_low=0,
            init_range_high=n_elements,
            gene_space=list(range(n_elements)),
            random_mutation_min_val=0,
            random_mutation_max_val=n_elements-1,
            random_seed=self.random_seed,
            **search_kwargs,
        )
        ga_instance.run()
        self.logger.log_residual_data(data_dicts=data_dicts)
        if print_result is True:
            data = sorted(data_dicts, key=lambda xx: xx["score_dict"]["rate"], reverse=True)[0]
            rate, symbols = data["score_dict"]["rate"], data["elements"]
            print(f"Best Structure of Run {self.random_seed}:")
            print_search_results(symbols=symbols, rate=rate)
        # Run the Genetic Algorithm.
        return data_dicts


# -------------------------------------------------------------------------------------
# GET OPTMIMIZATION METHOD
# -------------------------------------------------------------------------------------

def get_optimization_method(
        method:str,
        element_pool:list,
        template_atoms_surf:Atoms,
        reaction_rate_func:callable,
        filename:str,
        pth_header:str=None,
        random_seed = 42,
        log_interval = 100, 
        database_kwargs = {},
        reaction_rate_kwargs = {},
    ) -> tuple:
    optimizers = {
        "RandomSearching":RandomSearcher,
        "DualAnnealing":DualAnnealing,
        "GeneticAlgorithm":GeneticAlgorithm,
        "ScikitOptimizer":ScikitOptimizer
    }

    search_kwargs_tot = {
        "GeneticAlgorithm":
            {
                "sol_per_pop": 100,
                "keep_parents": 2,
                "num_parents_mating": 50,
                "mutation_percent_genes": 30,
                "parent_selection_type": "tournament", # sss | rws | rank | random | tournament
                "crossover_type": "uniform", # single_point | two_points | uniform
                "mutation_type": "swap", # random | swap | inversion | scramble
            },
        "RandomSearching":
            {},
        "DualAnnealing":
            {},
        "ScikitOptimizer":
            {
                #"n_initial_points": int(n_samples / 10),
                "base_estimator": "GBRT", # GP | RF | ET | GBRT
                "acq_func": "LCB", # EI | LCB | PI | EIps | PIps | gp_hedge
                "acq_optimizer": "auto", # auto | sampling | lbfgs
            }
    }

    if method in optimizers:
        optimizer = optimizers[method](
            reaction_rate_func=reaction_rate_func,
            element_pool=element_pool,
            template_atoms_surf=template_atoms_surf,
            filename=filename,
            pth_header=pth_header,
            random_seed=random_seed,
            log_interval=log_interval,
            database_kwargs=database_kwargs,
            reaction_rate_kwargs=reaction_rate_kwargs
        )
        return optimizer, search_kwargs_tot[method]
    else:
        raise Exception(f"{method} has not been implemented, please choose from {optimizers.keys()}")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------