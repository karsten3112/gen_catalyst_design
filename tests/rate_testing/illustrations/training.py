from gen_catalyst_design.discrete_space_diffusion import (
    DiffusionModel, GNNLogitPredictor, CosineScheduler, ExponentialScheduler,
    AbsorbingStateNoiser, UniformTransitionsNoiser, NoneConditioning
)

from gen_catalyst_design.utils import (
    setup_trainer_and_logger
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)

import wandb
from ase.io import read, write
from ase_ml_models.databases import get_atoms_list_from_db
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.db import connect
import random
import torch
import os

def main():
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    noiser_type = "Absorbing" # Absorbing | Uniform
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    add_active_site_connectivity = True
    miller_index = "100"
    
    
    if "Absorbing" in noiser_type:
        element_pool = ["(X)"] + element_pool

    if noiser_type == "Absorbing":
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool,
            active_site_freezing=0.0,
            label_smoothing=0.0
        )
    elif noiser_type == "Uniform":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool
        )
    else:
        raise Exception(f"noiser of type {noiser_type} is not implemented")

    scheduler = ExponentialScheduler(
        beta_max=5e-2, 
        beta_min=1e-4,
        time_sample_method="stratified"
    )


    conditioning = NoneConditioning()
    hidden_dim = 32
    #Applying none as conditioning making it unconditional 
    logit_predictor = GNNLogitPredictor(
        num_elements=len(element_pool),
        conditioning_dim=conditioning.embedding_dim,
        embedding_dim=hidden_dim,
        time_embedding_dim=hidden_dim,
        hidden_rep_dim=hidden_dim,
        n_interaction_blocks=3,
        message_dim=hidden_dim
    )


    #Apply guidance
    #epoch=epoch=313-train=train_loss=0.8425
    #path_model = "nn_rate_predictor/3mpl2/checkpoints/last.ckpt"
    #rate_guidance = ReactionRateModule.load_from_checkpoint(path_model)
        
    #Also setting drop_prob to zero for training purposes.
    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        logit_predictor=logit_predictor,
        conditioning=conditioning,
        drop_prob=0.0,
        use_x0_reparam=True,
        d3pm_auxillary_weight=0.1,
        auxillary_rate_weight=None,
        num_kl_div_estimates=1,
        lr=1e-3
    )
    
    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../class_1.traj", index=":"),
        element_pool=element_pool,
        condition_key=None,
        add_active_site_connectivity=add_active_site_connectivity,
        batch_size=40
    )
  
    trainer_kwargs={
        "max_epochs":-1,
        "log_every_n_steps":1, 
        "enable_progress_bar":True, 
        "enable_model_summary":True
    }

    logger_kwargs = {}

    trainer = setup_trainer_and_logger(
        project_name="diff_model_uncond",
        model_name="mean_class_1",
        accelerator="cpu",
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
    


if __name__ == "__main__":
    main()