from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from .calculators import GCNNCalculator, GraphCalculator
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from catalyst_opt_tools.utilities import preprocess_features
from ase_ml_models.databases import get_atoms_list_from_db
from ase.db import connect
import numpy as np
import yaml
import torch
import wandb
import os

# -------------------------------------------------------------------------------------
# SETUP TRAINER
# -------------------------------------------------------------------------------------

class DumpCheckpointDataCallback(Callback):
    def __init__(self, dict_key="hyper_params", filename="checkpoint_data.yaml"):
        self.dict_key = dict_key
        self.filename = filename

    def on_train_end(self, trainer, pl_module):
        # Find the best / last checkpoint PL saved
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = trainer.checkpoint_callback.last_model_path

        if ckpt_path == "":
            print("No checkpoint found to extract custom data.")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Extract the dictionary you stored earlier
        if self.dict_key not in ckpt:
            print(f"Key '{self.dict_key}' not found in checkpoint.")
            return

        data = ckpt[self.dict_key]

        # Write YAML
        run_dir = trainer.logger.experiment.dir
        out_path = os.path.join(run_dir, self.filename)

        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

        # Upload to W&B
        wandb.save(out_path)


def setup_trainer_and_logger(
        project_name:str,
        model_name:str=None,
        patience:int=50, 
        gradient_clip_val:float=1.0,
        checkpoint_dir:str="checkpoints",
        accelerator:str="gpu",
        pth_header:str=None,
        trainer_kwargs:dict={},
        logger_kwargs:dict={}
    ) -> Trainer:

    if not os.path.exists(pth_header):
        os.makedirs(pth_header)

    if model_name is None:
        model_name = "model"
        if pth_header is not None:
            filenames = os.listdir(pth_header)
        else:
            filenames = os.listdir()
        model_num = 0
        for file in filenames:
            if pth_header is not None:
                file_search = os.path.join(pth_header,file)
            else:
                file_search = file    
            if os.path.isdir(file_search) and model_name in file:
                model_num+=1
        model_num+=1
        model_name = f"{model_name}_{model_num:03d}"
        if pth_header is not None:
            save_dir = os.path.join(pth_header, model_name)
        else:
            save_dir = model_name
    else:
        if pth_header is not None:
            save_dir = os.path.join(pth_header,model_name)
        else:
            save_dir = model_name
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logger = WandbLogger(
        project=project_name,
        name=model_name,
        save_dir=save_dir,
        **logger_kwargs
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, checkpoint_dir),
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

    hyper_params_log = DumpCheckpointDataCallback(
        dict_key="diffusion_parameters",
        filename="model_parameters.yaml"
    )

    trainer = Trainer(
        logger=logger,
        default_root_dir=save_dir,
        callbacks=[checkpoint_callback, early_stopping, hyper_params_log],
        gradient_clip_val=gradient_clip_val,
        devices=1,
        accelerator=accelerator,
        **trainer_kwargs
    )
    return trainer

def get_atoms_from_template_db(
    db_filename:str,
    pth_header:str=None
):
    """
    Get atoms from template database.
    """
    # Read atoms objects from templates database.
    if pth_header is not None:
        db_filename = os.path.join(pth_header, db_filename)
    db_ase = connect(db_filename)
    atoms_list = get_atoms_list_from_db(db_ase=db_ase)
    # Get number of atoms in the surface.
    n_atoms_surf = len([
        atoms for atoms in atoms_list if atoms.info["species"] == "clean"
    ][0])
    # Return the list of atoms objects.
    return atoms_list, n_atoms_surf


def get_features_bulk_and_gas(
        bulk_filename:str="features_bulk.yaml", 
        gas_filename:str="features_gas.yaml", 
        pth_header:str=None
        ) -> tuple:
        """
        Get features for bulk and gas phase.
        """
        if pth_header is not None:
            bulk_filename = os.path.join(pth_header, bulk_filename)
            gas_filename = os.path.join(pth_header, gas_filename)
        # Read features from yaml files.
        with open(bulk_filename, "r") as fileobj:
            features_bulk = yaml.safe_load(fileobj)

        with open(gas_filename, "r") as fileobj:
            features_gas = yaml.safe_load(fileobj)
        # Preprocess features.
        features_bulk = preprocess_features(features_dict=features_bulk)
        features_gas = preprocess_features(features_dict=features_gas)
        # Return parameters.
        return features_bulk, features_gas


def get_calculator(model, miller_index):
    if model == "WWL-GPR":
        calculator = GraphCalculator(
             miller_index=miller_index,
             kernel="GPR",
             S2=0.0
        )
        train_kwargs = {}
    elif model == "GCNN":
        calculator = GCNNCalculator(
            miller_index=miller_index
        )
        network_hyperparams = {"hidden_dim":128,
            "n_conv_layers": 4,
            "n_lin_layers": 2,
            "conv_type": "ARMAConv",
            "dropout":0.0,
            "activation": torch.nn.functional.relu,
            #"use_batch_norm":False,
            #"use_residual":False
        }
        train_kwargs = {
                    "early_stop":True,
                    "num_epochs":500,
                    "lr": 1e-4,
                    "batch_size": 16,
                    "target": "E_form",
                    "val_split": 0.1,
                    "early_stopping_patience": 100,
                    "early_stopping_delta": 1e-4,
                    "weight_decay":0.0,
                    "hyperparams":network_hyperparams
        }
    else:
         raise Exception(f"calculator of type: {model} has not been implemented yet")
    return calculator, train_kwargs



def get_periodic_surface(miller_index:str, vacuum:float=10.0, n_layers_z:int=4, a_lat:float=None) -> tuple:
    if miller_index == "100":
        from ase.build import fcc100
        atoms_periodic = fcc100(symbol="Au", size=(3, 3, n_layers_z), vacuum=vacuum, a=a_lat)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "111":
        from ase.build import fcc111
        atoms_periodic = fcc111(symbol="Au", size=(3, 3, n_layers_z), vacuum=vacuum, a=a_lat)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "211":
        from ase.build import fcc211
        atoms_periodic = fcc211(symbol="Au", size=(6, 3, n_layers_z), vacuum=vacuum, a=a_lat)
        indices_site = [0, 1, 7, 10, 15, 16]
    # Highlight site atoms.
    for ii in indices_site:
        atoms_periodic[ii].symbol = "Cu"
    return atoms_periodic, indices_site

def exclude_species(
        elements:list,
        species_exclude:list
    ):
    result_list = []
    for element in elements:
        if element not in species_exclude:
            result_list.append(element)
    return result_list

def get_full_element_pool(
        species_exclude:list=["Mn", "Ga"]
    ):
    result_list = ["Rh", "Pt", "Pd", "Co", "Ga", "Cu", "Zn", "Au", "Ag", 'Mn', 'Fe', 'Os', 'Mo', 'Ir', 'Ru', "Ni"]
    if species_exclude is not None:
        result_list = exclude_species(
            elements=result_list,
            species_exclude=species_exclude
        )
    return result_list



def get_rate_ecdf(
        rate_distribution:np.array
    ):
    result = ecdf(rate_distribution)
    return result