from gen_catalyst_design.discrete_space_diffusion.diffusion import (
    setup_diffusion_model
)

from gen_catalyst_design.utils import (
    setup_trainer_and_logger,
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)
from distutils.util import strtobool
from ase.io import read
import argparse
import random
import torch
import wandb

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--scheduler",
    "-sched",
    type=str,
    required=False,
    default="ExponentialBetaScheduler",
)

parser.add_argument(
    "--noiser",
    "-noiser",
    type=str,
    required=False,
    default="AbsorbingStateNoiser",
)

parser.add_argument(
    "--drop_prob",
    "-p_drop",
    type=float,
    required=False,
    default=0.1,
)

parser.add_argument(
    "--model_name",
    "-m_name",
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
    #General settings
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    element_pool = ["Rh", "Cu", "Au", "Pd"]

    conditioning_kwargs = {
        "embedding_dim":64
    }

    scheduler_kwargs = {
        "time_sample_method":"stratified"
    }

    if parsed_args.scheduler == "ExponentialBetaScheduler":
        scheduler_kwargs.update({"beta_max":5e-2, "beta_min":1e-4})

    logit_predictor_kwargs = {
        "embedding_dim":64,
        "time_embedding_dim":64,
        "hidden_rep_dim":64,
        "message_dim":64,
        "n_interaction_blocks":5,

    }

    diff_model_kwargs = {
        "drop_prob":parsed_args.drop_prob,
        "lr":1e-3
    }

    #construct the diffusion model
    diff_model, condition_keys = setup_diffusion_model(
        element_pool=element_pool,
        noiser_type=parsed_args.noiser,
        scheduler_type=parsed_args.scheduler,
        add_rate_conditioning=True,
        add_e_form_conditioning=False,
        scheduler_kwargs=scheduler_kwargs,
        logit_predictor_kwargs=logit_predictor_kwargs,
        conditioning_kwargs=conditioning_kwargs,
        diff_model_kwargs=diff_model_kwargs
    )

    #Load in the data and setup training and validation loaders
    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../../tests/datasets/genetic_alg/no_duplicates.traj", index=":"),
        element_pool = element_pool,
        condition_keys=condition_keys,
        add_active_site_connectivity=False,
        use_fully_connected_graph=False,
        batch_size=40
    )
    
    #Trainer parameters
    trainer_kwargs={
        "max_epochs":2000,
        "log_every_n_steps":1, 
        "enable_progress_bar":True, 
        "enable_model_summary":True,
        "deterministic":True
    }

    #logger paraemters
    logger_kwargs = {}

    #construct the trainer for handling training process
    trainer = setup_trainer_and_logger(
        project_name="hyperparams",
        model_name=parsed_args.model_name,
        pth_header="trained_models",
        accelerator=parsed_args.device,
        trainer_kwargs=trainer_kwargs,
        logger_kwargs=logger_kwargs
    )

    #train the diffusion model
    trainer.fit(
        model=diff_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    wandb.finish()

if __name__ == "__main__":
    main()