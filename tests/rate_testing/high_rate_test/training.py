from gen_catalyst_design.discrete_space_diffusion import (
    RateClassEmbedder, DiffusionModel, ClassLabelEmbedder,
    DiscreteGNNDenoiser, CosineScheduler, ExponentialScheduler,
    AbsorbingStateNoiser, UniformTransitionsNoiser
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

def main():
    noiser_type = "Uniform" # Absorbing | Uniform
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    embedding_dim = 24
    
    
    if "Absorbing" in noiser_type:
        element_pool = ["(X)"] + element_pool

    if noiser_type == "Absorbing":
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool
        )
        auxillary_weight = 0.01
    elif noiser_type == "Uniform":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool
        )
        auxillary_weight = 0.0
    else:
        raise Exception(f"noiser of type {noiser_type} is not implemented")

    condition_embedder = ClassLabelEmbedder(num_labels=1, embedding_dim=embedding_dim)
    denoiser = DiscreteGNNDenoiser(
        element_pool=element_pool,
        cond_embedder=condition_embedder,
        message_dim=embedding_dim,
        hidden_dim_rep=embedding_dim
    )
        
    scheduler = CosineScheduler(beta_max=1e-1, beta_min=1e-3)

    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        denoiser=denoiser,
        drop_prob=0.10,
        use_x0_reparam=True,
        auxillary_weight=auxillary_weight
    )

    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("training_set.traj", index=":"),
        element_pool=element_pool,
        batch_size=40
    )

    trainer_kwargs={
        "max_epochs":-1,
        "log_every_n_steps":50, 
        "enable_progress_bar":True, 
        "enable_model_summary":True
    }

    logger_kwargs = {}

    trainer = setup_trainer_and_logger(
        model_name=noiser_type,
        trainer_kwargs=trainer_kwargs,
        logger_kwargs=logger_kwargs,
        patience=50,
    )

    trainer.fit(
        model=diff_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )






if __name__ == "__main__":
    main()