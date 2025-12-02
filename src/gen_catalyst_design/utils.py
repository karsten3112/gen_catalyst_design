from .discrete_space_diffusion import (
    DiscreteSpaceNoiser,
    ConditioningEmbedder,
)
from gen_catalyst_design.reaction_rates import ReactionMechanism
from ase_ml_models.pyg import get_edges_list_from_connectivity
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from torch_geometric.loader import DataLoader
from catalyst_opt_tools.utilities import preprocess_features, update_atoms_list
import yaml
import torch
import wandb
import torch.nn.functional as F
from ase.atoms import Atoms
import random
import os

# -------------------------------------------------------------------------------------
# SETUP NOISER
# -------------------------------------------------------------------------------------

def setup_noiser(noiser_type:str, noiser_params:dict={}) -> DiscreteSpaceNoiser:
    from gen_catalyst_design.discrete_space_diffusion import AbsorbingStateNoiser, UniformTransitionsNoiser
    noisers = {"AbsorbingStateNoiser": AbsorbingStateNoiser, "UniformTransitionsNoiser": UniformTransitionsNoiser}
    if noiser_type in noisers:
        return noisers[noiser_type](**noiser_params)
    else:
        raise NotImplementedError(f"Noiser of type {noiser_type} is not implemented")

# -------------------------------------------------------------------------------------
# SETUP SCHEDULER
# -------------------------------------------------------------------------------------

def setup_scheduler(scheduler_type:str, scheduler_params:dict={}):
    from gen_catalyst_design.discrete_space_diffusion import LinearScheduler, ExponentialScheduler, CosineScheduler
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
    from gen_catalyst_design.discrete_space_diffusion import DiscreteGNNDenoiser
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
    from gen_catalyst_design.discrete_space_diffusion import ClassLabelEmbedder, RateClassEmbedder, RateEmbedder
    cond_embs = {
        "ClassLabelEmbedder":ClassLabelEmbedder,
        "RateEmbedder":RateEmbedder,
        "RateClassEmbedder": RateClassEmbedder
    }
    if condition_type in cond_embs:
        return cond_embs[condition_type](**condition_params)
    else:
        raise NotImplementedError(f"Conditioning of type {condition_type} is not implemented")

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
        model_name:str=None,
        patience:int=10, 
        gradient_clip_val:float=2.0,
        checkpoint_dir:str="checkpoints",
        trainer_kwargs:dict={},
        logger_kwargs:dict={}
    ) -> Trainer:


    if model_name is None:
        model_name = "models"
        if not os.path.exists("models"):
            os.makedirs("models")
    else:
        if not os.path.exists(model_name):
            os.makedirs(model_name)
    
    existing_models = os.listdir(model_name)
    run_id = len(existing_models) + 1
    run_name = f"model_{run_id:03d}"

    out_dir = os.path.join(model_name, run_name)

    os.makedirs(out_dir)

    logger = WandbLogger(
        project=model_name,
        name=run_name,
        save_dir=out_dir,
        **logger_kwargs
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, checkpoint_dir),
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
        default_root_dir=out_dir,
        callbacks=[checkpoint_callback, early_stopping, hyper_params_log],
        gradient_clip_val=gradient_clip_val,
        **trainer_kwargs
    )
    return trainer


# -------------------------------------------------------------------------------------
# EMBEDDING METHODS
# -------------------------------------------------------------------------------------

def embed_elements_as_onehot(elements:list, element_pool:list):
    mapping_dict = {element:i for i, element in enumerate(element_pool)}
    return torch.stack([get_onehot(element=element, mapping_dict=mapping_dict) for element in elements])

def embed_cluster_as_onehots(atoms:Atoms, element_pool:list):
    elements = atoms.get_chemical_symbols()
    return embed_elements_as_onehot(elements=elements, element_pool=element_pool)

def get_onehot(element:str, mapping_dict:dict):
    onehot = F.one_hot(torch.tensor(mapping_dict[element]), len(mapping_dict))
    return onehot

def get_graph_from_datadict(datadict:dict, template_atoms:Atoms, element_pool:list, condition_key:str=None):
    from gen_catalyst_design.discrete_space_diffusion import Graph
    edges_list = get_edges_list_from_connectivity(template_atoms.info["connectivity"])
    edge_index = torch.tensor(edges_list, dtype=torch.long).reshape(2,-1)
    graph = Graph(
        x=embed_elements_as_onehot(elements=datadict["elements"], element_pool=element_pool),
        edge_index=edge_index,
        pos=torch.tensor(template_atoms.positions)
    )
    if condition_key is not None:
        if condition_key in datadict:
            graph.y = datadict[condition_key]
        else:
            raise Exception(f"condition key {condition_key} is not available in datadict, having: {datadict.keys()}")
    return graph

def get_dataset_from_datadicts(datadicts:list, template_atoms:Atoms, element_pool:list, condition_key:str=None):
    from gen_catalyst_design.discrete_space_diffusion import GraphDataset
    graph_list = [
        get_graph_from_datadict(
            datadict=datadict, 
            template_atoms=template_atoms, 
            element_pool=element_pool, 
            condition_key=condition_key
        )
        for datadict in datadicts
    ]
    return GraphDataset(graph_list=graph_list)


def get_dataloaders_from_datadicts(
        data_dicts:list, 
        element_pool:list,
        template_atoms:Atoms,
        batch_size:int=42,
        condition_key:str="class", 
        train_val_split:float=0.1,
        loader_kwargs:dict={} 
    ):
    random.shuffle(data_dicts)
    split_index = int((1-train_val_split)*len(data_dicts))
    train_dataset = get_dataset_from_datadicts(
        datadicts=data_dicts[:split_index],
        template_atoms=template_atoms,
        element_pool=element_pool,
        condition_key=condition_key
    )
    val_dataset = get_dataset_from_datadicts(
        datadicts=data_dicts[split_index:],
        template_atoms=template_atoms,
        element_pool=element_pool,
        condition_key=condition_key
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )
    return train_loader, val_loader

# -------------------------------------------------------------------------------------
# GET DATASET & DATALOADERS
# -------------------------------------------------------------------------------------

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
    from gen_catalyst_design.calculators import GCNNCalculator, GraphCalculator
    if model == "WWL-GPR":
        calculator = GraphCalculator(
             miller_index=miller_index,
             kernel="GPR"
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


def reaction_rate_of_RDS_from_symbols(
    reaction_mechanism:ReactionMechanism,
    symbols: list,
    template_atoms_list: list,
    features_bulk: dict,
    features_gas: dict,
    n_atoms_surf: int,
):
    """
    Get reaction rate of the RDS from the surface symbols.
    """
    # Update elements and features of atoms for predictions.
    update_atoms_list(
        atoms_list=template_atoms_list,
        features_bulk=features_bulk,
        features_gas=features_gas,
        symbols=symbols,
        n_atoms_surf=n_atoms_surf,
    )
    # Predict formation energies with a calculator.
    # Calculate reaction rate from reaction mechanism.
    score_dict = reaction_mechanism(atoms_list=template_atoms_list)
    return score_dict