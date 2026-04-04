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
    element_pool = get_full_element_pool() #['Ni', 'Cu', 'Rh', 'Ir', 'Pd', 'Pt', 'Au', 'Ag']#
    condition_keys = ["rate"]

    if noiser == "AbsorbingStateNoiser":
        element_pool = ["(X)"] + element_pool

    conditioning_kwargs = {
        "embedding_dim":64,
        "rate_min":-16.0,
        "rate_max":6.0
    }

    scheduler_kwargs = {
        "beta_max":5e-2, 
        "beta_min":1e-4,
        "time_sample_method":"stratified"
    }

    diff_model_kwargs = {
        "drop_prob":0.1,
        "lr":1e-4,
        #"d3pm_auxillary_weight":0.01
    }

    atoms_list = read("../../results/diffusion_model_parameters/datasets/random_search_2000.traj", index=":")

    filtered_atoms_list = atoms_list #filter_dataset(
    #    atoms_list=atoms_list,
    #    log_rate_offset=1e-3
    #)

    #Load in the data and setup training and validation loaders
    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=filtered_atoms_list,
        element_pool = element_pool,
        condition_keys=condition_keys,
        add_active_site_connectivity=False,
        use_fully_connected_graph=False,
        batch_size=20,
        graph_kwargs={"use_log":True}
    )

    #construct the diffusion model
    diff_model = setup_diffusion_model(
        element_pool=element_pool,
        noiser_type=noiser,
        scheduler_type=scheduler_type,
        condition_keys=condition_keys,
        scheduler_kwargs=scheduler_kwargs,
        conditioning_kwargs=conditioning_kwargs,
        diff_model_kwargs=diff_model_kwargs
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
        project_name="rate_new",
        model_name="rnd_search1",
        pth_header="rate_new",
        accelerator="cpu",
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


def filter_dataset(
        atoms_list:list,
        num_classes:int=100,
        max_samples_per_class:int=100,
        use_log:bool=True,
        log_rate_offset:float=1e-2
    ):

    rates = np.array([atoms.info["rate"] for atoms in atoms_list])
    if use_log:
        rates = np.log(rates)
        if log_rate_offset is not None:
            min_rate = np.log(np.exp(np.min(rates))+log_rate_offset)
    else:
        min_rate = np.min(rates)
    max_rate = np.max(rates)
    class_divisions = np.linspace(min_rate, max_rate, num_classes)
    class_indices = np.digitize(rates, class_divisions)
    filtered_atoms_list = []
    for idx in range(num_classes):
        indices = np.argwhere(class_indices==idx+1)
        n_samples_in_class, _ = indices.shape
        if n_samples_in_class > max_samples_per_class:
            store_indices = np.random.permutation(indices.squeeze())[:max_samples_per_class]
        else:
            store_indices = indices.squeeze(axis=-1)
        for store_index in store_indices:
            filtered_atoms_list.append(atoms_list[store_index])
    return filtered_atoms_list





if __name__ == "__main__":
    main()