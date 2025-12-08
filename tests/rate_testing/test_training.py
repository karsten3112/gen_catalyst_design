from gen_catalyst_design.db import Database, load_data_from_db
from gen_catalyst_design.discrete_space_diffusion import (
    RateClassEmbedder, DiffusionModel, ClassLabelEmbedder,
    DiscreteGNNDenoiser, CosineScheduler, ExponentialScheduler,
    AbsorbingStateNoiser, UniformTransitionsNoiser
)
from gen_catalyst_design.utils import (
    setup_trainer_and_logger
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_datadicts
)
import wandb
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
    #Setting up the diffusion model
    assign_no_class = True
    opt_methods = [
        #"random_search",
        "GeneticAlgorithm",
        #"DualAnnealing"
    ]

    schedulers = [
        CosineScheduler, 
        ExponentialScheduler
    ]

    beta_maxs = [1e-1] #1e-2

    noiser_type = "AbsorbingStateNoiser"
    
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    if "Absorbing" in noiser_type:
        element_pool = ["(X)"] + element_pool

    if noiser_type == "AbsorbingStateNoiser":
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool
        )
    elif noiser_type == "UniformTransitionsNoiser":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool
        )
    else:
        raise Exception(f"noiser of type {noiser_type} is not implemented")
    

    #Setting up the dataloaders
    miller_index = "100"
    run_ids = [0,1,2]
    rate_min = 1.0
    
    db = connect("../../databases/templates/100/100_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]

    dataloader_dict = {}
    num_classes = {}
    for opt_method in opt_methods:
        pth_header = f"../../results/{opt_method}/results/Rh_Cu_Au_Pd/miller_index_{miller_index}"
        datadicts = []
        for runid in run_ids:
            filename = f"runID_{runid}_results.db"
            database = Database.establish_connection(filename=filename, miller_index=miller_index, pth_header=pth_header)
            datadicts += load_data_from_db(database=database)
        filtered_dicts = filter_data_dicts(data_dicts=datadicts, rate_min=rate_min)
        rates = [data["rate"] for data in filtered_dicts]
        rate_max = np.max(rates)
        step = 2.5
        class_ranges = np.arange(rate_min, np.ceil(rate_max)+step, step)
        num_classes[opt_method] = len(class_ranges) - 1
        assign_rate_class(data_dicts=filtered_dicts, class_ranges=class_ranges, assign_no_class=assign_no_class)

        train_loader, val_loader = get_dataloaders_from_datadicts(
            data_dicts=filtered_dicts,
            element_pool=element_pool,
            template_atoms=template_atoms,
            batch_size=40
        )
        dataloader_dict[opt_method] = {"train_loader":train_loader, "val_loader":val_loader}

    diff_models = {}
    random_seed = 42
    for opt_method in opt_methods:
        #for scheduler in schedulers:
        #    for beta_max in beta_maxs:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        condition_embedder = ClassLabelEmbedder(num_labels=num_classes[opt_method], embedding_dim=8)
        denoiser = DiscreteGNNDenoiser(
            element_pool=element_pool,
            cond_embedder=condition_embedder,
            message_dim=8,
            hidden_dim_rep=8
            )
        
        scheduler = CosineScheduler(beta_min=1e-4, beta_max=1e-1)

        diff_model = DiffusionModel(
            element_pool=element_pool,
            scheduler=scheduler,
            noiser=noiser,
            denoiser=denoiser,
            drop_prob=0.10
        )
        diff_models[opt_method] = diff_model

    #Setting up trainer and logger
    trainer_kwargs={
        "max_epochs":-1,
        "log_every_n_steps":50, 
        "enable_progress_bar":True, 
        "enable_model_summary":True
    }

    logger_kwargs = {}
    for opt_method in dataloader_dict:
        dataloaders = dataloader_dict[opt_method]
        #Training the diffusion model
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        trainer = setup_trainer_and_logger(
            model_name=f"{opt_method}_training_absorb",
            trainer_kwargs=trainer_kwargs,
            logger_kwargs=logger_kwargs
        )
        trainer.fit(
            model=diff_models[opt_method],
            train_dataloaders=dataloaders["train_loader"],
            val_dataloaders=dataloaders["val_loader"]
        )
        wandb.finish()
    
def assign_rate_class(data_dicts:list, class_ranges:np.array, assign_no_class:bool=False):
    for datadict in data_dicts:
        rate = datadict["rate"]
        indices = np.argwhere(rate > class_ranges)
        if assign_no_class:
            datadict.update({"class":0})
        else:
            datadict.update({"class":len(indices)-1})
        

def filter_data_dicts(data_dicts:list, rate_min:float=None):
    if rate_min is None:
        return data_dicts
    else:
        filtered_datadicts = []
        for datadict in data_dicts:
            if datadict["rate"] < rate_min:
                pass
            else:
                filtered_datadicts.append(datadict)
        return filtered_datadicts
    
if __name__ == "__main__":
    main()
    