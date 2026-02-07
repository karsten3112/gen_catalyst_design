
from gen_catalyst_design.discrete_space_diffusion.Dataset import get_dataloaders_from_atoms_list
from gen_catalyst_design.discrete_space_diffusion.guidance import rateGNN, ReactionRateModule
from ase.io import read
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import random
import torch
import os

def main():
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    add_active_site_connectivity = True
    element_pool = ["(X)"] + ["Rh", "Cu", "Au", "Pd"]
    hidden_dim = 32

    reaction_rate_gnn = rateGNN(
        element_pool=element_pool,
        message_dim=hidden_dim,
        n_hidden_layers=3,
        hidden_dim_rep=hidden_dim,
    )

    reaction_rate_module = ReactionRateModule(
        reaction_rate_nn=reaction_rate_gnn,
        lr=1e-3
    )

    train_loader, test_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../high_rate_structs.traj", index=":"),
        element_pool=element_pool,
        condition_key="rate",
        add_active_site_connectivity=add_active_site_connectivity,
        do_initial_shuffling=True,
        batch_size=40
    )
    #train_data = train_loader.dataset
    #rates = torch.tensor([graph.y for graph in train_data])
    #mean_rate = torch.mean(rates)
    #var_rate = torch.sqrt(torch.var(rates))
    #for graph in train_data:
    #    graph.y-=mean_rate
    #    graph.y/=var_rate

    trainer_kwargs={
        "max_epochs":-1,
        "log_every_n_steps":1, 
        "enable_progress_bar":True, 
        "enable_model_summary":True
    }

    logger_kwargs = {}

    trainer = setup_trainer_and_logger(
        project_name="gnn_rate_pred",
        model_name="3mpl2",
        accelerator="cpu",
        trainer_kwargs=trainer_kwargs,
        logger_kwargs=logger_kwargs,
        gradient_clip_val=1.0,
        patience=30
    )

    trainer.fit(
        model=reaction_rate_module,
        train_dataloaders=train_loader
    )

def setup_trainer_and_logger(
        project_name:str,
        model_name:str=None,
        patience:int=10, 
        gradient_clip_val:float=2.0,
        checkpoint_dir:str="checkpoints",
        accelerator:str="gpu",
        trainer_kwargs:dict={},
        logger_kwargs:dict={}
    ) -> Trainer:

    
    if model_name is None:
        model_name = "model"
        filenames = os.listdir()
        model_num = 0
        for file in filenames:
            if os.path.isdir(file) and model_name in file:
                model_num+=1
        model_num+=1
        model_name = f"{model_name}_{model_num:03d}"
        os.makedirs(model_name)
    else:
        if not os.path.exists(model_name):
            os.makedirs(model_name)

    logger = WandbLogger(
        project=project_name,
        name=model_name,
        save_dir=model_name,
        **logger_kwargs
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(model_name, checkpoint_dir),
        monitor="train_loss",
        mode="min",
        save_top_k=1,      # keep best model
        save_last=True,    # also save last model
        filename="epoch={epoch}-train={train_loss:.4f}",
    )
    early_stopping = EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=patience,
        min_delta=0.0,
    )

    trainer = Trainer(
        logger=logger,
        default_root_dir=model_name,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=gradient_clip_val,
        devices=1,
        accelerator=accelerator,
        **trainer_kwargs
    )
    return trainer


if __name__ == "__main__":
    main()