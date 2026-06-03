
from gen_catalyst_design.discrete_space_diffusion.Dataset import get_train_val_dataloaders, get_train_val_atoms_list
from gen_catalyst_design.utils import get_features_bulk_and_gas, setup_trainer_and_logger
from gen_catalyst_design.optimization import setup_optimization_objective, evaluate_score_from_symbols, Logger
from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.db import Database, load_datadicts_from_db, load_atoms_list_from_db
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.stability import Stabilizer
from distutils.util import strtobool
from ase.io import read, write
from ase.atoms import Atoms
import wandb
import numpy as np
import argparse
import torch
import random
import os
# -------------------------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--pretrained_model_ckpt",
    "-model_ckpt",
    type=str,
    required=True,
    default=None
)

parser.add_argument(
    "--init_samples_traj",
    "-init_traj",
    type=str,
    required=True,
    default=None
)

parser.add_argument(
    "--pre_estim_samles_db",
    "-pre_sample_db",
    type=str,
    required=False,
    default=None
)

parser.add_argument(
    "--active_learning_loops",
    "-n_loops",
    type=int,
    required=False,
    default=5,
)

parser.add_argument(
    "--num_samples_per_loop",
    "-n_samples_per_loop",
    type=int,
    required=False,
    default=1000,
)

parser.add_argument(
    "--include_stability",
    "-inc_stab",
    type=fbool,
    required=False,
    default=False,
)

parser.add_argument(
    "--filter_top_percent",
    "-filter_top",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--random_seed",
    "-rnd_seed",
    type=int,
    required=False,
    default=42,
)

parser.add_argument(
    "--miller_index",
    "-m_index",
    type=str,
    required=False,
    default="100",
)

parser.add_argument(
    "--project_name",
    "-proj_name",
    type=str,
    required=False,
    default="active_learning",
)

parser.add_argument(
    "--outdir",
    "-out",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "--device",
    "-dev",
    type=str,
    required=False,
    default="cpu",
)

parsed_args = parser.parse_args()


def main():
    random_seed = parsed_args.random_seed
    num_samples_per_loop = parsed_args.num_samples_per_loop
    n_learning_loops = parsed_args.active_learning_loops
    filter_top_percent = parsed_args.filter_top_percent

    miller_index = parsed_args.miller_index
    include_stability = parsed_args.include_stability
    
    init_samples_traj = parsed_args.init_samples_traj
    init_diff_model_ckpt = parsed_args.pretrained_model_ckpt
    pre_estim_db = parsed_args.pre_estim_samles_db

    log_project_name = parsed_args.project_name
    outdir = parsed_args.outdir
    device = parsed_args.device

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


    reaction_mechanism, stabilizer, _ = setup_optimization_objective(
        miller_index=miller_index,
        template_type="surface" if include_stability else "cluster",
        database_pth_header="../../gen_catalyst_design/databases",
        yaml_files_header="../../gen_catalyst_design/yaml_files",
        include_stability=include_stability,
    )

    sampling_kwargs = {
        "temp":1.0,
        "batch_size":100
    }

    percentile_kwargs = {
        "lower_percent_lim":0.90,
        "upper_percent_lim":0.99,
        "num_conds":5
    }

    if pre_estim_db is not None:
        database = Database.establish_connection(
            filename=pre_estim_db
        )
        add_atoms_list = load_atoms_list_from_db(
            database=database,
            template_atoms_surf=reaction_mechanism.clean_surface
        )

    init_atoms_list = read(filename=init_samples_traj, index=":")

    run_active_learning(
        init_atoms_list=init_atoms_list,
        diff_model_ckpt=init_diff_model_ckpt,
        reaction_mechanism=reaction_mechanism,
        guidance_scale_dict={"rate":2.0},
        project_name=log_project_name,
        outdir=outdir,
        n_learning_loops=n_learning_loops,
        pre_estimated_samples=add_atoms_list,
        sampling_kwargs=sampling_kwargs,
        num_samples_per_iter=num_samples_per_loop,
        stabilizer=stabilizer,
        device=device,
        random_seed=random_seed,
        filter_samples_func=filter_top_percentile if filter_top_percent else None,
        filter_samples_kwargs={},
        percentile_kwargs=percentile_kwargs,
        use_log=True
    )

def run_active_learning(
        init_atoms_list:list,
        diff_model_ckpt:str,
        reaction_mechanism:ReactionMechanism,
        guidance_scale_dict:dict={"rate":2.0},
        project_name:str="active_learning",
        outdir:str=None,
        patience:int=50,
        n_learning_loops:int=5,
        train_val_split:float=0.2,
        do_initial_shuffling:bool=True,
        pre_estimated_samples:list=None,
        sampling_kwargs:dict={},
        num_samples_per_iter:int=1000,
        stabilizer:Stabilizer=None,
        device:str="cpu",
        random_seed:int=42,
        filter_samples_func:callable=None,
        filter_samples_kwargs:dict={},
        percentile_kwargs:dict={},
        use_log:bool=True
    ):

    train_atoms_list, val_atoms_list = get_train_val_atoms_list(
        atoms_list=init_atoms_list,
        train_val_split=train_val_split,
        do_initial_shuffling=do_initial_shuffling,
        random_seed=random_seed
    )

    loop_iter = 0
    while loop_iter < n_learning_loops:
        diff_model = DiffusionModel.load_from_checkpoint(diff_model_ckpt).to(device=torch.device(device))
        if loop_iter == 0 and pre_estimated_samples is not None:
            if filter_samples_func is not None:
                pre_estimated_samples = filter_samples_func(pre_estimated_samples, train_atoms_list+val_atoms_list, **filter_samples_kwargs)
            atoms_add_train, atoms_add_val = get_train_val_atoms_list(
                atoms_list=pre_estimated_samples,
                train_val_split=train_val_split,
                do_initial_shuffling=do_initial_shuffling,
                random_seed=random_seed
            )
            train_atoms_list+=atoms_add_train
            val_atoms_list+=atoms_add_val
        else:
            database = Database.establish_connection(
                filename=f"samples.db",
                pth_header=os.path.join(outdir, f"learn_iter_{loop_iter}"),
                database_kwargs={
                    "append":False, 
                    "template_atoms_surf":reaction_mechanism.clean_surface
                }
            )
            #filter_samples_kwargs.update({"atoms_list_orig":training_samples})
            new_samples = sample_and_evaluate_scores(
                diff_model=diff_model,
                reaction_mechanism=reaction_mechanism,
                database=database,
                sample_distribution=train_atoms_list+val_atoms_list,
                guidance_scale_dict=guidance_scale_dict,
                stabilizer=stabilizer,
                num_samples=num_samples_per_iter,
                diff_model_sampling_kwargs=sampling_kwargs,
                filter_samples=filter_samples_func,
                filter_kwargs=filter_samples_kwargs,
                percentile_kwargs=percentile_kwargs
            )
            
            atoms_add_train, atoms_add_val = get_train_val_atoms_list(
                atoms_list=new_samples,
                train_val_split=train_val_split,
                do_initial_shuffling=do_initial_shuffling,
                random_seed=random_seed
            )
            train_atoms_list+=atoms_add_train
            val_atoms_list+=atoms_add_val
        
        
        train_loader, val_loader = get_train_val_dataloaders(
            train_atoms_list=train_atoms_list,
            val_atoms_list=val_atoms_list,
            element_pool=diff_model.element_pool,
            condition_keys=list(guidance_scale_dict.keys()),
            batch_size=40,
            device=device,
            graph_kwargs={"use_log":use_log}
        )

        #Trainer parameters
        trainer_kwargs={
            "max_epochs":5000,
            "log_every_n_steps":1, 
            "enable_progress_bar":False, 
            "enable_model_summary":True,
            "deterministic":True
        }

        #logger parameters
        logger_kwargs = {}

        #construct the trainer for handling training process
        trainer = setup_trainer_and_logger(
            project_name=project_name,
            model_name=f"learn_iter_{loop_iter+1}",
            pth_header=outdir,
            accelerator=device,
            trainer_kwargs=trainer_kwargs,
            save_every_n_epochs=100,
            patience=patience,
            logger_kwargs=logger_kwargs
        )

        #train the diffusion model
        trainer.fit(
            model=diff_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        wandb.finish()

        diff_model_ckpt = get_next_diff_model_ckpt(
            model_name=f"learn_iter_{loop_iter+1}",
            pth_header=outdir
        )
        loop_iter+=1
    
    database = Database.establish_connection(
        filename=f"samples.db",
        pth_header=os.path.join(outdir, f"learn_iter_{loop_iter}"),
        database_kwargs={
            "append":False, 
            "template_atoms_surf":reaction_mechanism.clean_surface
        }
    )

    new_samples = sample_and_evaluate_scores(
        diff_model=diff_model,
        reaction_mechanism=reaction_mechanism,
        database=database,
        sample_distribution=train_atoms_list+val_atoms_list,
        guidance_scale_dict=guidance_scale_dict,
        stabilizer=stabilizer,
        num_samples=num_samples_per_iter,
        diff_model_sampling_kwargs=sampling_kwargs,
        percentile_kwargs=percentile_kwargs
    )
    
def get_samples_from_diff_model(
        diff_model:DiffusionModel,
        template_atoms:Atoms,
        sample_distribution:list,
        guidance_scale_dict:dict={"rate":2.0},
        condition_keys:list=["rate"],
        num_samples:int=1000,
        use_log:bool=True,
        sampling_kwargs:dict={},
        percentile_kwargs:dict={}
    ):

    conditioning_dicts = []

    if "rate" in condition_keys:
        rate_conditions = get_rate_conditions(
            atoms_list=sample_distribution,
            use_log=use_log,
            percentile_kwargs=percentile_kwargs
        )

        conditioning_dicts += get_rate_condition_dicts(
            rate_conditions=rate_conditions,
            num_samples=num_samples
        )

    if "e_form" in condition_keys:
        e_form_conditions = get_e_form_conditions()
        add_e_form_condition_to_dicts(
            conditioning_dicts=conditioning_dicts,
            e_form_conditions=e_form_conditions
        )

    samples = diff_model.sample(
        n_samples=num_samples,
        conditioning_dicts=conditioning_dicts,
        template_atoms=template_atoms,
        condition_keys=condition_keys,
        guidance_scale_dict=guidance_scale_dict,
        return_as_atoms_list=True,
        dataset_kwargs={"graph_kwargs":{"use_log":False if use_log else True}},
        **sampling_kwargs
    )
    return samples


def sample_and_evaluate_scores(
        diff_model:DiffusionModel,
        reaction_mechanism:ReactionMechanism,
        database:Database,
        sample_distribution:list,
        guidance_scale_dict:dict={"rate":2.0},
        stabilizer:Stabilizer=None,
        num_samples:int=1000,
        use_log_diff_model:bool=True,
        diff_model_sampling_kwargs:dict={},
        filter_samples:callable=None,    
        filter_kwargs:dict={},
        percentile_kwargs:dict={}
    ):

    samples = get_samples_from_diff_model(
        diff_model=diff_model,
        template_atoms=reaction_mechanism.clean_surface,
        sample_distribution=sample_distribution,
        guidance_scale_dict=guidance_scale_dict,
        condition_keys=list(guidance_scale_dict.keys()),
        num_samples=num_samples,
        use_log=use_log_diff_model,
        sampling_kwargs=diff_model_sampling_kwargs,
        percentile_kwargs=percentile_kwargs
    )

    logger = Logger(
        database=database,
        log_interval=100
    )
    result_samples = []
    for atoms in samples:
        result_dict = evaluate_score_from_symbols(
            symbols=atoms[0].get_chemical_symbols(),
            reaction_mechanism=reaction_mechanism,
            stabilizer=stabilizer,
            logger=logger,
            objective_key="datadict"
        )
        for key in result_dict["score_dict"]:
            atoms[0].info = reaction_mechanism.clean_surface.info.copy()
            atoms[0].info[key] = result_dict["score_dict"][key]
        result_samples.append(atoms[0])
    
    logger.write_data_to_file()
    database.close_connection()
    if filter_samples is not None:
        filtered_samples = filter_samples(
            result_samples, sample_distribution, **filter_kwargs
        )
        return filtered_samples
    else:
        return result_samples


def get_rate_condition_dicts(
        rate_conditions:np.array,
        num_samples:int,
    ):
    result_dicts = []
    n_samples_per_rate_cond = int(num_samples/len(rate_conditions))
    for rate_cond in rate_conditions:
        condition_dict = {"rate":rate_cond}
        result_dicts += [condition_dict]*n_samples_per_rate_cond
    return result_dicts


def get_rate_conditions(
        atoms_list:list,
        use_log:bool=True,
        percentiles:np.array=None,
        percentile_kwargs:dict={}
    ):
    rates = np.array([atoms.info["rate"] for atoms in atoms_list])
    if use_log:
        rates = np.log10(rates)

    if percentiles is None:
        percentiles = np.linspace(
            percentile_kwargs.pop("lower_percent_lim",0.8), 
            percentile_kwargs.pop("upper_percent_lim",0.95), 
            percentile_kwargs.pop("num_conds", 5)
        )
    conditions = np.array([np.round(np.percentile(rates, percentile*100), 2) for percentile in percentiles])
    return conditions

def add_e_form_condition_to_dicts(
        conditioning_dicts:list,
        e_form_conditions:np.array,
    ):
    pass

def get_e_form_conditions() -> np.array:
    pass

def get_next_diff_model_ckpt(
        model_name:str,
        model_type:str="best",
        pth_header:str=None,
    ):
    file_kws = {
        "best": "best",
        "last": "last"
    }
    ckpt_dir = os.path.join(model_name, "checkpoints")
    if pth_header is not None:
        ckpt_dir = os.path.join(pth_header, ckpt_dir)
    ckpt_files = os.listdir(ckpt_dir)
    kw = file_kws[model_type]
    for ckpt_file in ckpt_files:
        if kw in ckpt_file:
            return os.path.join(ckpt_dir, ckpt_file)


def filter_top_percentile(
        atoms_list_add:list,
        atoms_list_orig:list,
        percentile:float=95.0,
        condition_key:str="rate",
        use_log:bool=True
    ):
    values_orig = np.array([atoms.info[condition_key] for atoms in atoms_list_orig])
    if use_log:
        values_orig = np.log10(values_orig)
    
    values_add = np.array([atoms.info[condition_key] for atoms in atoms_list_add])
    if use_log:
        values_add = np.log10(values_add)

    threshold = np.percentile(values_orig, percentile)
    print(threshold)
    #print(values_orig)
    print(values_add)
    indices = np.argwhere(values_add >= threshold).flatten()
    print(indices)
    filtered_atoms_list = [atoms_list_add[index] for index in indices]
    return filtered_atoms_list


if __name__ == "__main__":
    main()