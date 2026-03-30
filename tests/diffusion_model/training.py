from gen_catalyst_design.discrete_space_diffusion.diffusion import (
    setup_diffusion_model
)

from gen_catalyst_design.utils import (
    setup_trainer_and_logger,
    get_full_element_pool
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)

import wandb
from ase.io import read
import numpy as np
import random
import torch

def main():
    #Control
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    #diffusion model parameters
    noiser = "AbsorbingStateNoiser" # absorbing | uniform
    scheduler_type = "ExponentialBetaScheduler" # exponential
    element_pool = ['Ni', 'Cu', 'Rh', 'Ir', 'Pd', 'Pt', 'Au', 'Ag']#get_full_element_pool(["Mn", "Ga"])

    if noiser == "AbsorbingStateNoiser":
        element_pool = ["(X)"] + element_pool

    add_rate_conditioning = True
    add_e_form_conditioning = True

    conditioning_kwargs = {
        "embedding_dim":64
    }

    scheduler_kwargs = {
        "beta_max":5e-2, 
        "beta_min":1e-4,
        "time_sample_method":"stratified"
    }

    logit_predictor_kwargs = {
        "embedding_dim":64,
        "time_embedding_dim":64,
        "hidden_rep_dim":64,
        "message_dim":64,
        "n_interaction_blocks":5,

    }

    diff_model_kwargs = {
        "drop_prob":0.1,
        "lr":1e-4,
        #"d3pm_auxillary_weight":0.01
    }

    #construct the diffusion model
    diff_model, condition_keys = setup_diffusion_model(
        element_pool=element_pool,
        noiser_type=noiser,
        scheduler_type=scheduler_type,
        add_rate_conditioning=add_rate_conditioning,
        add_e_form_conditioning=add_e_form_conditioning,
        scheduler_kwargs=scheduler_kwargs,
        logit_predictor_kwargs=logit_predictor_kwargs,
        conditioning_kwargs=conditioning_kwargs,
        diff_model_kwargs=diff_model_kwargs
    )
  
    #Load in the data and setup training and validation loaders
    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../../results/reconstruction_check/chgnet_result_all_fcc.traj", index=":"),
        element_pool = element_pool,
        condition_keys=condition_keys,
        add_active_site_connectivity=False,
        use_fully_connected_graph=False,
        batch_size=40,
        graph_kwargs={"use_log":True}
    )
    #for batch in train_loader:
    #    print(batch.e_form)
    #exit()
    #Trainer parameters
    trainer_kwargs={
        "max_epochs":5000,
        "log_every_n_steps":1, 
        "enable_progress_bar":True, 
        "enable_model_summary":True,
        "deterministic":True
    }

    #logger paraemters
    logger_kwargs = {}

    #construct the trainer for handling training process
    trainer = setup_trainer_and_logger(
        project_name="rate_eform_testing",
        model_name="test_debug_cond",
        pth_header="rate_eform_testing",
        accelerator="gpu",
        trainer_kwargs=trainer_kwargs,
        logger_kwargs=logger_kwargs,
        patience=5000
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