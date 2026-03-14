# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------
import time
from .db import Database
from .reaction_rates import ReactionMechanism
from .stability import Stabilizer
from .utils import get_features_bulk_and_gas, get_calculator, get_atoms_from_template_db
from chgnet.model.dynamics import CHGNetCalculator
import os
#from catalyst_opt_tools.optimization import print_search_progress, print_search_results
#from skopt.optimizer import base_minimize
#from scipy.optimize import dual_annealing
#from skopt.space import Categorical
#from skopt.utils import use_named_args
#from pygad import GA

# -------------------------------------------------------------------------------------
# LOGGER CLASS
# -------------------------------------------------------------------------------------

class Logger:
    def __init__(
            self,
            database:Database=None,
            log_interval:int=100,
            match_log_interval_gen_iter:bool=True
        ):
        self.database = database
        self.log_interval = log_interval
        self.match_log_interval_gen_iter = match_log_interval_gen_iter
        self.stored_datadicts = []
        self.buffer_datadicts = []
        self.initial_log_performed = False
        self.times_logged = 0
        self.n_obj_func_calls = 0
        self.time_statistics = {}

    def reset_count_stats(self, log_interval:int):
        self.log_interval = log_interval
        self.initial_log_performed = False
        self.times_logged = 0
        self.n_obj_func_calls = 0

    def store_datadict(self, datadict):
        if self.match_log_interval_gen_iter:
            datadict.update({"gen_iter":self.times_logged})
        self.buffer_datadicts.append(datadict)
        if len(self.buffer_datadicts) == self.log_interval:
            self.write_data_to_file()

    def time_function_call(self, timing_label:str, function:callable, function_kwargs):
        t_init = time.time()
        result = function(**function_kwargs)
        t_elapsed = time.time() - t_init
        if timing_label in self.time_statistics:
            self.time_statistics[timing_label].append(t_elapsed)
        else:
            self.time_statistics[timing_label] = [t_elapsed]
        return result

    def write_data_to_file(self):
        self.stored_datadicts+=self.buffer_datadicts
        if self.database is not None:
            self.database.write_data_to_tables(data_dicts=self.buffer_datadicts)
        self.buffer_datadicts = []
        self.times_logged+=1


def evaluate_score_from_symbols(
        symbols:list,
        reaction_mechanism:ReactionMechanism,
        logger:Logger,
        stabilizer:Stabilizer=None,
        add_time_stats:bool=False,
        objective_key:str="rate",
        ):
        if add_time_stats:
            score_dict = logger.time_function_call(
                "rate_eval", 
                reaction_mechanism.get_rate_of_RDS_from_symbols,
                {"symbols":symbols}
            )
        else:
            score_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
                symbols=symbols
            )
        if stabilizer is not None:
            if add_time_stats:
                add_dict = logger.time_function_call(
                    "stability_eval",
                    stabilizer.get_formation_energy_from_symbols,
                    {"symbols":symbols}
                )
            else:
                add_dict = stabilizer.get_formation_energy_from_symbols(
                    symbols=symbols
                )
            score_dict.update(add_dict)
        datadict = {"elements":symbols, "score_dict":score_dict}
        logger.store_datadict(datadict=datadict)
        logger.n_obj_func_calls+=1
        return get_score_from_obj_key(datadict=datadict, objective_key=objective_key)


def get_score_from_obj_key(datadict:dict, objective_key:str="rate"):
    if objective_key == "rate":
        return datadict["score_dict"]["rate"]
    elif objective_key == "stability":
        return datadict["score_dict"]["e_form"]
    elif objective_key == "mixture":
        raise NotImplementedError("Not implemented yet for having objective for both rate and e_form")
    elif objective_key == "both":
        return datadict["score_dict"]["rate"], datadict["score_dict"]["e_form"]
    else:
        raise Exception(f"Other score-type of {objective_key} is not defined")

def setup_optimization_objective(
        miller_index:str,
        template_type:str,
        database_pth_header:str,
        yaml_files_header:str,
        include_stability:bool=False,
        calculator_kwargs:dict={},
        stability_kwargs:dict={},
        reaction_mechanism_kwargs:dict={}
    ):

    features_bulk, features_gas = get_features_bulk_and_gas(
        pth_header=os.path.join(yaml_files_header, "features")
    )
    
    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(
        model=calculator_kwargs.pop("model", "WWL-GPR"), 
        miller_index=miller_index
    )
    
    #Train calculator on database
    calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT_all.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header=os.path.join(database_pth_header, "DFT_database"),
         train_kwargs=train_kwargs
    )
    
    #get template atoms list from database
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=os.path.join(database_pth_header, f"{template_type}_templates")
    )
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        template_atoms_list=template_atoms_list,
        calculator=calculator,
        features_bulk=features_bulk,
        features_gas=features_gas,
        mechanism_pth_header=os.path.join(yaml_files_header, "reaction_mechanism")
    )

    if include_stability and template_type == "surface":
        if template_type != "surface":
            stabilizer = None
            print(f"stability is set to be included, but cannot be estimated based on template type: {template_type}")
        else:
            stabilizer = Stabilizer(
                calculator=stability_kwargs.pop("calculator", CHGNetCalculator()),
                template_atoms=reaction_mechanism.clean_surface,
                ref_energy_file=stability_kwargs.pop("ref_energies_yaml", "chgnet_ref_energies.yaml"),
                ref_energy_pth_header=os.path.join(yaml_files_header, "reference_energies"),
                fmax=stability_kwargs.pop("fmax",0.05)
            )
    else:
        stabilizer = None
    return reaction_mechanism, stabilizer, template_atoms_list