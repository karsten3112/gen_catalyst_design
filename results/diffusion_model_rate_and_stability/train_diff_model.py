from gen_catalyst_design.discrete_space_diffusion import (
    DiffusionModel, UniformTransitionsNoiser, AbsorbingStateNoiser,
    LinearBetaScheduler, ExponentialBetaScheduler, CosineScheduler, LinearAlphaScheduler,
    RateClassConditioning, RateScalarConditioning, NoneConditioning, EformConditioning,
    RateMantissaConditioning
)


from gen_catalyst_design.utils import (
    setup_trainer_and_logger,
    get_full_element_pool
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)
from distutils.util import strtobool
from ase.io import read
import numpy as np
import argparse
import random
import torch
import wandb

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--data_traj_file",
    "-data_traj",
    type=str,
    required=True,
    default="",
)


parser.add_argument(
    "--element_pool",
    "-elems",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "--scheduler",
    "-sched",
    type=str,
    required=False,
    default="ExponentialBetaScheduler",
)

parser.add_argument(
    "--beta_max",
    "-beta_max",
    type=float,
    required=False,
    default=1.0,
)

parser.add_argument(
    "--beta_min",
    "-beta_min",
    type=float,
    required=False,
    default=1e-4,
)

parser.add_argument(
    "--noiser",
    "-noiser",
    type=str,
    required=False,
    default="AbsorbingStateNoiser",
)

parser.add_argument(
    "--rate_conditioning",
    "-rate_cond",
    type=str,
    required=False,
    default="RateScalarConditioning",
)

parser.add_argument(
    "--e_form_conditioning",
    "-e_form_cond",
    type=str,
    required=False,
    default="EformConditioning",
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
    "--d3pm_aux",
    "-d3pm_aux",
    type=float,
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

parser.add_argument(
    "--max_epochs",
    "-m_epochs",
    type=int,
    required=False,
    default=1000,
)

parser.add_argument(
    "--patience",
    "-pat",
    type=int,
    required=False,
    default=None,
)

parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    required=False,
    default=1e-3,
)

parser.add_argument(
    "--log_every_n_epochs",
    "-n_epoch_log",
    type=int,
    required=False,
    default=200,
)

parser.add_argument(
    "--outdir",
    "-out",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "--log_project_name",
    "-log_proj_name",
    type=str,
    required=False,
    default="diffusion_model_full",
)

parser.add_argument(
    "--use_log",
    "-log",
    type=fbool,
    required=False,
    default=True,
)

parsed_args = parser.parse_args()


def main():
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    noiser_type = parsed_args.noiser
    scheduler_type = parsed_args.scheduler
    drop_prob = parsed_args.drop_prob
    device = parsed_args.device
    model_name = parsed_args.model_name
    d3pm_aux_weight = parsed_args.d3pm_aux
    outdir = parsed_args.outdir
    log_project_name = parsed_args.log_project_name
    max_epochs = parsed_args.max_epochs
    patience = parsed_args.patience
    lr = parsed_args.learning_rate
    log_every_n_epochs = parsed_args.log_every_n_epochs
    
    if parsed_args.element_pool is None:
        element_pool = get_full_element_pool()
    else:
        element_pool = parsed_args.element_pool.split(",")

    use_log = parsed_args.use_log
    dataset = read(parsed_args.data_traj_file, index=":")

    #add absorbing state token if noiser is absorbing and not already in element pool
    if noiser_type == "AbsorbingStateNoiser" and "(X)" not in element_pool:
        element_pool = ["(X)"] + element_pool

    #Setup noiser
    noiser_kwargs = {}

    noiser = setup_noiser(
        noiser_type=noiser_type,
        element_pool=element_pool,
        noiser_kwargs=noiser_kwargs
    )

    #Setup scheduler
    scheduler_kwargs = {
        "time_sample_method":"stratified",
        "beta_max":parsed_args.beta_max,
        "beta_min":parsed_args.beta_min
    }

    scheduler = setup_scheduler(
        scheduler_type=scheduler_type,
        scheduler_kwargs=scheduler_kwargs
    )

    #Setup rate conditioning
    condition_kwargs = {
        "embedding_dim":64
    }

    rate_conditioning = setup_rate_conditioning(
        condition_type=parsed_args.rate_conditioning,
        atoms_list=dataset,
        use_log=use_log,
        condition_kwargs=condition_kwargs
    )

    #Setup e-form conditioning
    e_form_conditioning = setup_e_form_conditioning(
        condition_type=parsed_args.e_form_conditioning,
        condition_kwargs=condition_kwargs
    )

    #Assemble diffusion model
    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        rate_conditioning=rate_conditioning,
        e_form_conditioning=e_form_conditioning, 
        drop_prob=drop_prob,
        lr=lr,
        weight_decay=0.0,
        d3pm_auxillary_weight=d3pm_aux_weight
    )

    #construct dataloaders for training
    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=dataset,
        element_pool=element_pool,
        batch_size=40,
        condition_keys=["rate", "e_form"],
        train_val_split=0.2,
        random_seed=random_seed,
        device=device,
        graph_kwargs={"use_log":use_log}
    )

    #Trainer parameters
    trainer_kwargs={
        "max_epochs":max_epochs,
        "log_every_n_steps":1, 
        "enable_progress_bar":False, 
        "enable_model_summary":True,
        "deterministic":True
    }

    #logger paraemters
    logger_kwargs = {}

    #construct the trainer for handling training process
    trainer = setup_trainer_and_logger(
        project_name=log_project_name,
        model_name=model_name,
        pth_header=outdir,
        accelerator=device,
        trainer_kwargs=trainer_kwargs,
        save_every_n_epochs=log_every_n_epochs,
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

def setup_rate_conditioning(
        condition_type:str,
        atoms_list:list=None,
        use_log:bool=True,
        condition_kwargs:dict={},
    ):
    if condition_type == "RateScalarConditioning":
        condition = RateScalarConditioning(
            embedding_dim=condition_kwargs.pop("embedding_dim", 64)
        )
        return condition
    elif condition_type == "RateMantissaConditioning":
        conditioning = RateMantissaConditioning(
            embedding_dim=condition_kwargs.pop("embedding_dim", 64)
        )
        return conditioning
    elif condition_type == "RateClassConditioning":
        rates = np.array([atoms.info["rate"] for atoms in atoms_list])
        if use_log:
            rates = np.log10(rates)
        min_rate, max_rate = np.floor(np.min(rates)), np.ceil(np.max(rates))
        condition = RateClassConditioning(
            rate_min=min_rate,
            rate_max=max_rate,
            embedding_dim=condition_kwargs.pop("embedding_dim", 64),
            num_classes=condition_kwargs.pop("num_classes", 20)
        )
        return condition
    elif condition_type == "NoneConditioning":
        conditioning = NoneConditioning(
            embedding_dim=condition_kwargs.pop("embedding_dim", 64)
        )
        return conditioning
    else:
        raise Exception(f"condition of type: {condition_type} is not implemented")


def setup_e_form_conditioning(
        condition_type:str,
        condition_kwargs:dict={}
    ):
    if condition_type == "EformConditioning":
        conditioning = EformConditioning(
            embedding_dim=condition_kwargs.pop("embedding_dim", 64)
        )
        return conditioning
    elif condition_type == "NoneConditioning":
        conditioning = NoneConditioning(
            embedding_dim=condition_kwargs.pop("embedding_dim", 64)
        )
        return conditioning

def setup_noiser(
        noiser_type:str,
        element_pool:list,
        noiser_kwargs:dict={}    
    ):
    if noiser_type == "AbsorbingStateNoiser":
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool,
            **noiser_kwargs
        )
        return noiser
    elif noiser_type == "UniformTransitionsNoiser":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool,
            **noiser_kwargs
        )
        return noiser
    else:
        raise Exception(f"Noiser of type: {noiser_type} is not implemented")
    pass


def setup_scheduler(
        scheduler_type:str,
        scheduler_kwargs:dict={}
    ):
    if scheduler_type == "LinearBetaScheduler":
        scheduler = LinearBetaScheduler(
            **scheduler_kwargs
        )
        return scheduler
    elif scheduler_type == "ExponentialBetaScheduler":
        scheduler = ExponentialBetaScheduler(
            **scheduler_kwargs
        )
        return scheduler
    elif scheduler_type == "CosineScheduler":
        scheduler = CosineScheduler(
            **scheduler_kwargs
        )
        return scheduler
    elif scheduler_type == "LinearAlphaScheduler":
        scheduler = LinearAlphaScheduler(
            **scheduler_kwargs
        )
        return scheduler
    else:
        raise Exception(f"Scheduler of type: {scheduler} is not implemented")


if __name__ == "__main__":
    main()