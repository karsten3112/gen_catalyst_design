from gen_catalyst_design.discrete_space_diffusion import (
    DiscreteTimeScheduler, LinearScheduler, CosineScheduler, ExponentialScheduler,
    DiscreteSpaceNoiser, AbsorbingStateNoiser, UniformTransitionsNoiser,
    ConditioningEmbedder, ClassLabelEmbedder, RateEmbedder,
    DiscreteSpaceDenoiser, DiscreteGNNDenoiser,
    DiffusionModel,
)

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from distutils.util import strtobool
import argparse
import torch

from .reaction_rate_calculation import (
    get_calculator, 
    get_features_bulk_and_gas, 
    get_atoms_from_template_db, 
    reaction_rate_of_RDS_from_symbols
)

# -------------------------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))


parser.add_argument(
    "--miller_index",
    "-m_index",
    type=str,
    required=False,
    default="100",
)

parser.add_argument(
    "--scheduler",
    "-scheduler",
    type=str,
    required=False,
    default="CosineScheduler",
)

parser.add_argument(
    "--noiser",
    "-noiser",
    type=str,
    required=False,
    default="AbsorbingStateNoiser",
)

parser.add_argument(
    "--denoiser",
    "-denoiser",
    type=str,
    required=False,
    default="DiscreteGNNDenoiser",
)

parser.add_argument(
    "--conditioning",
    "-cond",
    type=str,
    required=False,
    default="ClassLabelEmbedder",
)

parser.add_argument(
    "--element_pool",
    "-elem_pool",
    type=str,
    required=False,
    default="Au,Cu,Pd,Rh,Ni,Ga",
)

parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    required=False,
    default=1e-4,
)

parser.add_argument(
    "--save_dir",
    "-dir",
    type=str,
    required=False,
    default="",
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    miller_index = parsed_args.miller_index
    element_pool = parsed_args.element_pool.split(",")

    #Setup the scheduler
    scheduler_type = parsed_args.scheduler
    scheduler_params = {
        "beta_max":torch.tensor(1e-1),
        "beta_min":torch.tensor(1e-4)
    }
    scheduler = setup_scheduler(
        scheduler_type=scheduler_type, 
        scheduler_params=scheduler_params
    )

    #Setup the Noiser
    noiser_type = parsed_args.noiser
    if "Absorbing" in noiser_type:
        element_pool = ["(X)"] + element_pool
    noiser_params = {
        "element_pool":element_pool
    }
    noiser = setup_noiser(
        noiser_type=noiser_type, 
        noiser_params=noiser_params
    )

    #Setup the conditioner
    condition_type = parsed_args.conditioning
    condition_params = {}
    condition_embedder = setup_condition_embedder(
        condition_type=condition_type, 
        condition_params=condition_params
    )

    #Setup the denoiser
    denoiser_type = parsed_args.denoiser
    denoiser_params = {
        "input_dim":len(element_pool),
        "n_hidden_message_layers":1,
        "message_dim":8,
    }
    denoiser = setup_denoiser(denoiser_type=denoiser_type, denoiser_params=denoiser_params, condition_embedder=condition_embedder)

    #Construct the diffusion model
    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        denoiser=denoiser,
        lr=parsed_args.learning_rate
    )

    #Setup the logger
    logger = WandbLogger(
        name="diffusion_model",
        save_dir=parsed_args.save_dir
    )

    #Setup the trainer for doing the training
    
    trainer = setup_trainer(
        logger=logger,
    )
    
    #Setup dataloaders
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"databases/templates/{miller_index}"
    )

    train_loader, val_loader = setup_dataloaders()
    #Train the diffusion model
    
    trainer.fit(
        model=diff_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    

# -------------------------------------------------------------------------------------
# SETUP NOISER
# -------------------------------------------------------------------------------------

def setup_noiser(noiser_type:str, noiser_params:dict={}) -> DiscreteSpaceNoiser:
    noisers = {"AbsorbingStateNoiser": AbsorbingStateNoiser, "UniformTransitionsNoiser": UniformTransitionsNoiser}
    if noiser_type in noisers:
        return noisers[noiser_type](**noiser_params)
    else:
        raise NotImplementedError(f"Noiser of type {noiser_type} is not implemented")

# -------------------------------------------------------------------------------------
# SETUP SCHEDULER
# -------------------------------------------------------------------------------------

def setup_scheduler(scheduler_type:str, scheduler_params:dict={}):
    schedulers = {
        "LinearScheduler":LinearScheduler,
        "ExponentialScheduler":ExponentialScheduler,
        "CosineScheduler":CosineScheduler
    }
    if scheduler_type in schedulers:
        return schedulers[scheduler_type](**scheduler_params)
    else:
        raise NotImplementedError(f"Scheduler of type {scheduler_type} is not implemented")

# -------------------------------------------------------------------------------------
# SETUP DENOISER
# -------------------------------------------------------------------------------------

def setup_denoiser(denoiser_type:str, condition_embedder:ConditioningEmbedder, denoiser_params:dict={}):
    denoisers = {
        "DiscreteGNNDenoiser":DiscreteGNNDenoiser
    }
    if denoiser_type in denoisers:
        return denoisers[denoiser_type](**denoiser_params, cond_embedder=condition_embedder)
    else:
        raise NotImplementedError(f"Denoiser of type {denoiser_type} is not implemented")

# -------------------------------------------------------------------------------------
# SETUP CONDITION EMBEDDER
# -------------------------------------------------------------------------------------

def setup_condition_embedder(condition_type:str, condition_params:dict={}) -> ConditioningEmbedder:
    cond_embs = {
        "ClassLabelEmbedder":ClassLabelEmbedder,
        "RateEmbedder":RateEmbedder
    }
    if condition_type in cond_embs:
        return cond_embs[condition_type](**condition_params)
    else:
        raise NotImplementedError(f"Conditioning of type {condition_type} is not implemented")

# -------------------------------------------------------------------------------------
# SETUP TRAINER
# -------------------------------------------------------------------------------------

def setup_trainer(logger:WandbLogger, patience:int=10, gradient_clip_val:float=2.0) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,      # keep best model
        save_last=True,    # also save last model
        filename="epoch={epoch}-val={val_loss:.4f}",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=0.0,
    )
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=gradient_clip_val,

    )
    return trainer

# -------------------------------------------------------------------------------------
# SETUP DATALOADERS
# -------------------------------------------------------------------------------------

def setup_dataloaders():
    pass
    



# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------