from gen_catalyst_design.discrete_space_diffusion import (
    DiffusionModel, MPNNLogitPredictor, CosineScheduler, ExponentialBetaScheduler,
    AbsorbingStateNoiser, UniformTransitionsNoiser, NoneConditioning, RateConditioning
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
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    noiser_type = "Absorbing" # Absorbing | Uniform
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    ckpt_file = None#"5mpl_final_no_lr_sched/checkpoints/epoch=epoch=224-val=val_loss=0.8073.ckpt"
    
    
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

    scheduler = ExponentialBetaScheduler(
        beta_max=5e-2, 
        beta_min=1e-4,
        time_sample_method="stratified"
    )

    #NoneConditioning()
    hidden_dim = 64
    conditioning = RateConditioning(embedding_dim=hidden_dim)
    #Applying none as conditioning making it unconditional 
    logit_predictor = MPNNLogitPredictor(
        num_elements=len(element_pool),
        conditioning_dim=conditioning.embedding_dim,
        embedding_dim=hidden_dim,
        time_embedding_dim=32,
        hidden_rep_dim=hidden_dim,
        n_interaction_blocks=5,
        message_dim=hidden_dim
    )


    #Apply guidance
    #epoch=epoch=313-train=train_loss=0.8425
    #path_model = "nn_rate_predictor/3mpl2/checkpoints/last.ckpt"
    #rate_guidance = ReactionRateModule.load_from_checkpoint(path_model)
        
    #Also setting drop_prob to zero for training purposes.
    if ckpt_file is None:
        diff_model = DiffusionModel(
            element_pool=element_pool,
            scheduler=scheduler,
            noiser=noiser,
            logit_predictor=logit_predictor,
            rate_conditioning=conditioning,
            drop_prob=0.1,
            use_x0_reparam=True,
            d3pm_auxillary_weight=None,
            auxillary_rate_weight=None,
            num_kl_div_estimates=1,
            lr=1e-3
        )
    else:
        diff_model = DiffusionModel.load_from_checkpoint(ckpt_file).to("cpu")

    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../no_duplicates.traj", index=":"),
        element_pool=element_pool,
        condition_keys=["rate"],
        add_active_site_connectivity=False,
        use_fully_connected_graph=False,
        batch_size=40
    )
  
    trainer_kwargs={
        "max_epochs":2000,
        "log_every_n_steps":1, 
        "enable_progress_bar":True, 
        "enable_model_summary":True,
        "deterministic":True
    }

    logger_kwargs = {}

    trainer = setup_trainer_and_logger(
        project_name="reproduce",
        model_name="fix_1",
        accelerator="cpu",
        trainer_kwargs=trainer_kwargs,
        logger_kwargs=logger_kwargs,
        gradient_clip_val=1.0,
        patience=50
    )

    trainer.fit(
        model=diff_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_file
    )
    wandb.finish()
    


if __name__ == "__main__":
    main()