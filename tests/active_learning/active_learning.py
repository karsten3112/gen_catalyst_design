
from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.utils import get_features_bulk_and_gas, setup_trainer_and_logger
from gen_catalyst_design.calculators import GraphCalculator
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.discrete_space_diffusion.Dataset import get_dataloaders_from_atoms_list
import wandb
from ase.atoms import Atoms
from ase.io import read, write
from distutils.util import strtobool
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
    required=False,
    default="init_model/checkpoints/epoch=epoch=72-val=val_loss=0.8990.ckpt",
)

parser.add_argument(
    "--active_learning_loops",
    "-n_loops",
    type=int,
    required=False,
    default=2,
)

parser.add_argument(
    "--num_samples",
    "-n_samples",
    type=int,
    required=False,
    default=10,
)

parser.add_argument(
    "--keep_method",
    "-k_mth",
    type=str,
    required=False,
    default="top_k_samples",
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


parsed_args = parser.parse_args()


def main():
    num_samples = parsed_args.num_samples
    keep_method = parsed_args.keep_method
    n_learning_loops = parsed_args.active_learning_loops
    miller_index = parsed_args.miller_index

    random.seed(parsed_args.random_seed)
    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed_all(parsed_args.random_seed)

    yaml_file_pth = "../../yaml_files"
    db_file_pth = "../../databases"
    reaction_mechanism = setup_rate_calculation(
        features_pth=os.path.join(yaml_file_pth, "features"),
        db_train_pth_header=os.path.join(db_file_pth,"DFT_database"),
        mechanism_pth_header=os.path.join(yaml_file_pth, "reaction_mechanism"),
        template_db_pth=os.path.join(db_file_pth, "templates", miller_index),
        template_db_name=f"{miller_index}_templates.db",
        miller_index=miller_index
    )

    trainer_kwargs={
        "max_epochs": 5,
        "log_every_n_steps":1, 
        "enable_progress_bar":True, 
        "enable_model_summary":True
    }

    sampling_kwargs = {
        "conditioning_key":"rate",
        "guidance_scale":2.0,
        "temp":1.0
    }

    run_active_learning(
        init_training_atoms_list=read("no_duplicates.traj", index=":"),
        diff_model_ckpt=parsed_args.pretrained_model_ckpt,
        reaction_mechanism=reaction_mechanism,
        pre_estimated_samples=None,
        n_learning_loops=n_learning_loops,
        num_samples=num_samples,
        trainer_kwargs=trainer_kwargs,
        sampling_kwargs=sampling_kwargs
    )



def run_active_learning(
        init_training_atoms_list:list,
        diff_model_ckpt:str,
        reaction_mechanism:ReactionMechanism,
        pre_estimated_samples:list=None,
        n_learning_loops:int=5, 
        num_samples:int=600, 
        project_name:str="active_learning_test",
        trainer_kwargs:dict={},
        logger_kwargs:dict={},
        sampling_kwargs:dict={}
    ):
    loop_iter = 0
    training_samples = init_training_atoms_list
    while loop_iter < n_learning_loops:
        diff_model = DiffusionModel.load_from_checkpoint(diff_model_ckpt).to(device=torch.device("cuda"))
        #diff_model = diff_model.to(device=torch.device("cpu"))
        if loop_iter == 0 and pre_estimated_samples is not None:
            training_samples+=pre_estimated_samples
        else:
            samples = sample_conditionally(
                diffusion_model=diff_model,
                training_atoms_list=training_samples,
                reaction_mechanism=reaction_mechanism,
                num_samples=num_samples,
                **sampling_kwargs
            )
            estimated_samples = filter_top_k_rate_samples(
                sampled_atoms_list=samples
            )
            write(f"samples_iter_{loop_iter}.traj", samples)
            training_samples+=estimated_samples 
        
        train_loader, val_loader = get_dataloaders_from_atoms_list(
            atoms_list=training_samples,
            element_pool=diff_model.element_pool,
            condition_key="rate",
            add_active_site_connectivity=False,
            use_fully_connected_graph=False,
            batch_size=40
        )

        trainer = setup_trainer_and_logger(
            project_name=project_name,
            model_name=f"iter_{loop_iter}",
            accelerator="gpu",
            trainer_kwargs=trainer_kwargs,
            logger_kwargs=logger_kwargs,
            gradient_clip_val=1.0,
            patience=50
        )
        trainer.fit(
            model=diff_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        wandb.finish()
        diff_model_ckpt = get_next_diff_model_ckpt(
            model_name=f"iter_{loop_iter}"
        )
        loop_iter+=1


def get_next_diff_model_ckpt(
        model_name:str,
        model_type:str="best"
    ):
    file_kws = {
        "best": "epoch",
        "last": "last"
    }
    ckpt_dir = os.path.join(model_name, "checkpoints")
    ckpt_files = os.listdir(ckpt_dir)
    kw = file_kws[model_type]
    for ckpt_file in ckpt_files:
        if kw in ckpt_file:
            return os.path.join(ckpt_dir, ckpt_file)


def sample_conditionally(
        diffusion_model:DiffusionModel,
        training_atoms_list:list,
        reaction_mechanism:ReactionMechanism,
        num_samples:int=1000,
        rate_span:float=0.2,
        num_conds:int=3,
        conditioning_key:str="rate",
        guidance_scale:float=2.0,
        temp:float=1.0
    ):

    if conditioning_key != "rate":
        raise NotImplementedError("Not currently implemented for other conditionings")
    
    rates = torch.tensor([atoms.info["rate"] for atoms in training_atoms_list])
    max_rate = torch.max(rates)
    conditionings = torch.linspace(max_rate-rate_span*max_rate, max_rate, num_conds)
    n_samples_per_cond = int(num_samples/num_conds)
    conditioning_dicts = []
    for condition in conditionings:
        conditioning_dicts+=[{"rate": condition.item()} for _ in range(n_samples_per_cond)]
    if len(conditioning_dicts) < num_samples:
        conditioning_dicts+=[{"rate": conditionings[-1].item()} for _ in range(num_samples - len(conditioning_dicts))]
    samples = diffusion_model.sample(
        n_samples=num_samples,
        template_atoms=reaction_mechanism.clean_surface,
        condition_key=conditioning_key,
        conditioning_dicts=conditioning_dicts,
        guidance_scale=guidance_scale,
        return_as_atoms_list=True,
        batch_size=50,
        log_all_timesteps=False,
        temp=temp
    )
    atoms_list = [sample[0] for sample in samples]
    result_atoms_list = [atoms for atoms in atoms_list if "O" not in atoms.symbols]
    for atoms in result_atoms_list:
        result_dict = reaction_mechanism.get_rate_RDS_from_atoms(atoms=atoms)
        atoms.info = reaction_mechanism.clean_surface.info.copy()
        atoms.info["rate"] = result_dict["rate"]
    return result_atoms_list
    

def setup_rate_calculation(
        features_pth:str,
        db_train_pth_header:str,
        mechanism_pth_header:str,
        template_db_name:str,
        template_db_pth:str=None,
        miller_index:str="100",
    ) -> ReactionMechanism:

    calculator = GraphCalculator(
        miller_index=miller_index,
        kernel="GPR"
    )
    features_bulk, features_gas = get_features_bulk_and_gas(
        pth_header=features_pth
    )
    calculator.train_model_from_db(
         db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
         features_bulk=features_bulk, 
         features_gas=features_gas, 
         db_pth_header=db_train_pth_header,
         train_kwargs={}
    )
    reaction_mechanism = ReactionMechanism(
        calculator=calculator,
        mechanism_pth_header=mechanism_pth_header,
        features_bulk=features_bulk,
        features_gas=features_gas
    )
    reaction_mechanism.set_template_atoms_list(
        db_filename=template_db_name,
        pth_header=template_db_pth
    )
    return reaction_mechanism


def filter_top_k_rate_samples(
        sampled_atoms_list:list,
        top_percentage:float=0.1
    ):
    #Could be smart to check structurally if they are the same and exclude if they are.
    tot_atoms_list =  sampled_atoms_list
    rates = torch.tensor([atoms.info["rate"] for atoms in tot_atoms_list])
    max_rate = torch.max(rates)
    indices = torch.argwhere(rates >= (1.0-top_percentage)*max_rate)
    filtered_atoms = [tot_atoms_list[index.item()] for index in indices]
    return filtered_atoms


if __name__ == "__main__":
    main()