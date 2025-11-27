from ase_ml_models.graph import graph_train, graph_predict, graph_preprocess
from ase_ml_models.pyg import pyg_predict, pyg_train
from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.pyg import create_pyg_dataset, PyGRegression
from catalyst_opt_tools.utilities import preprocess_features, update_atoms_list
import yaml
from mikimoto import units
from mikimoto.microkinetics import Species
from mikimoto.thermodynamics import ThermoNASA7
from sklearn.preprocessing import MinMaxScaler
from ase.db import connect
import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Calculator:
    def __init__(
            self, 
            miller_index:str
        ):
        self.model_params = None
        self.model = None
        self.bep_relation = self.get_bep_relation(miller_index=miller_index)

    def get_bep_relation(self, miller_index):
        if miller_index == "100":
            e_act_RDS = lambda delta_h: 0.789 + 0.624 * delta_h # [eV]
        elif miller_index == "111":
            e_act_RDS = lambda delta_h: 1.220 + 0.655 * delta_h # [eV]
        else:
            raise Exception(f"Following miller indices: {miller_index} have no bep-relation")
        return e_act_RDS

    def predict_with_model(self, atoms_list:list):
        raise Exception("Must be implemtented by sub-class")
    

    def get_lowest_e_form(self, atoms_list):
        e_form_dict = {"(X)": 0.0}
        bond_info_dict = {}
        for atoms in atoms_list:
            species = atoms.info["species"].replace("**", "(X)").replace("*", "(X)")
            e_form = atoms.info["E_form"]
            if species not in e_form_dict or e_form < e_form_dict[species]:
                e_form_dict[species] = e_form
                bond_info_dict[species] = atoms.info["bond_info"]
        return e_form_dict, bond_info_dict
    
    def __call__(self, atoms_list:list):
        atoms_list = self.predict_with_model(atoms_list=atoms_list)
        e_form_dict, bond_info_dict = self.get_lowest_e_form(atoms_list=atoms_list)
        return {"e_form_info":e_form_dict, "bond_info":bond_info_dict}

    def train_model_from_db(self, 
                            db_filename:str, 
                            features_bulk:dict,
                            features_gas:dict,
                            db_pth_header:str=None,
                            train_kwargs:dict={}
        ):
        if db_pth_header is not None:
            db_filename = os.path.join(db_pth_header, db_filename)
        db_ase = connect(name=db_filename)
        atoms_list = get_atoms_list_from_db(db_ase=db_ase)
        self.train_model(atoms_train=atoms_list, 
                         features_bulk=features_bulk, 
                         features_gas=features_gas,
                         train_kwargs=train_kwargs
                         )
    
    def train_model(self, atoms_train:list, features_bulk:dict, features_gas:dict, train_kwargs:dict):
        raise Exception("Must be implemtented by sub-class")


class GraphCalculator(Calculator):
    def __init__(self, miller_index, kernel, 
                 preproc:object = None
                 ):
        super().__init__(miller_index)
        self.model_params = {
                "target": "E_form",
                "model_name": kernel, #"KRR", #Are we sure about this?
                "kwargs_kernel": {"length_scale": 30},
                "kwargs_model": {"alpha": 1e-4},
        }
        self.preproc_params = {
                "node_weight_dict": {"A0": 1.00, "S1": 0.80, "S2": 0.20},
                "edge_weight_dict": {"AA": 0.50, "AS": 1.00, "SS": 0.50},
                "preproc": preproc,
        }


    def train_model(self, atoms_train:list, features_bulk:dict, features_gas:dict, train_kwargs:dict={}):
        update_atoms_list(atoms_list=atoms_train, 
                          features_bulk=features_bulk, 
                          features_gas=features_gas
        )
        graph_preprocess(
            atoms_list=atoms_train,
            **self.preproc_params,
        )
        self.model = graph_train(
            atoms_train=atoms_train,
            **self.model_params,
        )
        

    def predict_with_model(self, atoms_list: list):
        """
        Get formation energies with a graph model.
        """
        # Preprocess the data.
        graph_preprocess(
            atoms_list=atoms_list,
            **self.preproc_params,
        )
        # Predict test data.
        y_pred = graph_predict(
            atoms_test=atoms_list,
            model=self.model,
            **self.model_params,
        )
        # Update formation energies.
        for atoms, e_form in zip(atoms_list, y_pred):
            atoms.info["E_form"] = e_form
        return atoms_list


class GCNNCalculator(Calculator):
    def __init__(self, miller_index):
        super().__init__(miller_index)

    def train_model(self, atoms_train, features_bulk, features_gas, train_kwargs:dict={}):
        update_atoms_list(atoms_list=atoms_train, 
                          features_bulk=features_bulk, 
                          features_gas=features_gas
        )
        if "early_stop" in train_kwargs:
            early_stop = train_kwargs["early_stop"]
            train_kwargs.pop("early_stop")
            if early_stop:
                self.model, _, _, _ = pyg_train_with_early_stopping(atoms_train=atoms_train, **train_kwargs)
            else:
                self.model = pyg_train(atoms_train=atoms_train, **train_kwargs)
        else:
            self.model = pyg_train(atoms_train=atoms_train, **train_kwargs)

    
    def predict_with_model(self, atoms_list):
        y_pred = pyg_predict(atoms_test=atoms_list, model=self.model)
        for atoms, e_form in zip(atoms_list, y_pred):
            atoms.info["E_form"] = e_form
        return atoms_list


def pyg_train_with_early_stopping(
    atoms_train: list,
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    target: str = "E_form",
    hyperparams: dict = {},
    val_split: float = 0.1,
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 1e-4,
    weight_decay:float=0.0,
    shuffle:bool=True,
    print_results:bool=False,
    kwargs_scheduler:dict={}
):
    """
    Train a PyTorch Geometric model with optional validation and early stopping.
    """
    # Create tran and val datasets and dataloaders from ASE atoms list.
    from sklearn.model_selection import train_test_split
    atoms_train, atoms_val = train_test_split(atoms_train, test_size=val_split)
    dataset_train = create_pyg_dataset(atoms_train, target=target)
    dataset_val = create_pyg_dataset(atoms_val, target=target)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle)
    # Initialize the PyTorch Geometric model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = hyperparams if hyperparams else {}
    model = PyGRegression(
        num_node_features=dataset_train[0].num_node_features,
        **hyperparams,
        seed=None #Random seed does not work in here?
    ).to(device)
    # Train the PyTorch Geometric model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        **(kwargs_scheduler if kwargs_scheduler else {}),
    )
    loss_fn = torch.nn.MSELoss()
    print("Training PyG model with validation and early stopping.")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    lr_state = []
    for epoch in range(num_epochs):
        # Train.
        model.train()
        total_loss = 0
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(dataset_train)
        scheduler.step(train_loss)
        lr_state.append(optimizer.state_dict()["param_groups"][0]["lr"])
        # Validation.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader_val:
                batch = batch.to(device)
                out = model(batch)
                val_loss += loss_fn(out, batch.y).item() * batch.num_graphs
        val_loss /= len(dataset_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if print_results:
            print(f"Epoch {epoch+1:4d}: Train = {train_loss:.4f} Val = {val_loss:.4f}")
        # Early stopping check.
        if val_loss + early_stopping_delta < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    # Return the trained PyTorch Geometric model.
    return model, torch.tensor(train_losses), torch.tensor(val_losses), lr_state

