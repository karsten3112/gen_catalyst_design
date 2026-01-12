from gen_catalyst_design.discrete_space_diffusion import (
    RateClassEmbedder, DiffusionModel, ClassLabelEmbedder, ActiveSiteConditioning,
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
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    noiser_type = "Absorbing" # Absorbing | Uniform
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    hidden_dim = 64
    
    if "Absorbing" in noiser_type:
        element_pool = ["(X)"] + element_pool

    if noiser_type == "Absorbing":
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool
        )
    elif noiser_type == "Uniform":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool
        )
    else:
        raise Exception(f"noiser of type {noiser_type} is not implemented")

    scheduler = ExponentialScheduler(beta_max=2e-2, beta_min=1e-4)
    cond_embedder = ActiveSiteConditioning(element_pool=element_pool)
    #Applying none as conditioning making it unconditional 
    denoiser = DiscreteGNNDenoiser(
        element_pool=element_pool,
        cond_embedder=cond_embedder,
        message_dim=hidden_dim,
        n_hidden_layers=1,
        hidden_dim_rep=hidden_dim,
        time_embedding_dim=hidden_dim
    )
        
    #Also setting drop_prob to zero for training purposes.
    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        denoiser=denoiser,
        drop_prob=0.1,
        use_x0_reparam=True,
        auxillary_weight=0.1,
        auxillary_elem_weight=None,
        num_sample_estimate=5
    )

    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../high_rate_structs.traj", index=":"),
        element_pool=element_pool,
        condition_key="active_site_conf",
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
        project_name="active_site_cond",
        #model_name=noiser_type,
        trainer_kwargs=trainer_kwargs,
        logger_kwargs=logger_kwargs,
        gradient_clip_val=1.0,
        patience=30
    )

    trainer.fit(
        model=diff_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )






if __name__ == "__main__":
    main()