from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from ase.constraints import FixAtoms
import numpy as np
from ase_ml_models.pyg import get_edges_list_from_connectivity
import torch.nn.functional as F
from ase.atoms import Atoms
from ase.atom import Atom
import torch
import random



class Graph(Data):
    def __init__(
            self,
            x = None, 
            edge_index = None, 
            edge_attr = None,
            rate = None,
            e_form = None, 
            pos = None,
            active_sites = None,
            active_site_dists = None,
            **kwargs
        ):
        super().__init__(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            pos=pos,
            active_sites = active_sites,
            active_site_dists = active_site_dists,
            rate = rate,
            e_form = e_form,
            **kwargs
        )
        
    
    def to_elems(self, element_pool:list):
        indices = torch.argmax(self.x, dim=-1)
        return [element_pool[index] for index in indices]

    def to_atoms(self, element_pool:list):
        elements = self.to_elems(element_pool=element_pool)
        updated_elements = ["O" if elem == "(X)" else elem for elem in elements]
        atom_list = []
        for element, position in zip(updated_elements, self.pos):
            if position.device != "cpu":
                atom = Atom(symbol=element, position=position.cpu().numpy())
            else:
                atom = Atom(symbol=element, position=position.numpy())
            atom_list.append(atom)
        atoms = Atoms(atom_list)
        atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'O']))
        return atoms

    def update_x_from_elements(self, elements:list):
        x = embed_elements_as_onehot(elements=elements, element_pool=self.element_pool)
        self.x = x


class GraphDataset(Dataset):
    def __init__(self, graph_list, transform = None):
        super().__init__(transform=transform)
        self.graph_list = graph_list

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]

    def update_representation(self, new_repr, unique_batch_indices):
        for graph, new_x in zip(self[unique_batch_indices], new_repr):
            graph.x = new_x

    def to_atoms(self, element_pool):
        atoms_list = [graph.to_atoms(element_pool) for graph in self.graph_list]
        return atoms_list
    
    def to_elements(self, element_pool):
        elements_list = [graph.to_elems(element_pool) for graph in self.graph_list]
        return elements_list


def get_elements_from_onehots(x:torch.tensor, element_pool:list):
    indices = torch.argmax(x, dim=-1)
    if "(X)" in element_pool:
        elements = []
        for index in indices:
            if element_pool[index] == "(X)":
                elements.append("O")
            else:
                elements.append(element_pool[index])
        return elements
    else:
        return [element_pool[index] for index in indices]

def embed_elements_as_onehot(elements:list, element_pool:list, device:str=None):
    mapping_dict = {element:i for i, element in enumerate(element_pool)}
    return torch.stack([get_onehot(element=element, mapping_dict=mapping_dict, device=device) for element in elements])

def embed_cluster_as_onehots(atoms:Atoms, element_pool:list, device:str=None):
    elements = atoms.get_chemical_symbols()
    if "(X)" in element_pool:
        elements = ['(X)' if elem == 'O' else elem for elem in elements]
    return embed_elements_as_onehot(elements=elements, element_pool=element_pool, device=device)

def get_onehot(element:str, mapping_dict:dict, device:str=None):
    onehot = F.one_hot(torch.tensor(mapping_dict[element], device=device), len(mapping_dict))
    return onehot

def add_site_connections(connectivity, site_indices):
    for i in site_indices:
        for j in site_indices:
            entry = connectivity[i,j]
            if entry == 0 and i != j:
                connectivity[i,j] += 1

def get_active_site_dists(atoms:Atoms, site_indices:list):
    positions_all = atoms.positions
    positions_site = positions_all[site_indices]
    diff = positions_all[:, None, :] - positions_site[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    sorted_dists = np.sort(dists, axis=-1)[:, ::-1].copy()
    return sorted_dists

def get_graph_from_atoms(
        atoms:Atoms,
        element_pool:list,
        condition_keys:list=None,
        use_log:bool=True,
        include_recon_label:bool=False,
        device:str=None,
    ):
    x = embed_cluster_as_onehots(atoms=atoms, element_pool=element_pool, device=device)
    connectivity = atoms.info["connectivity"]
    site_indices = atoms.info["indices_site"]
    
    active_site_dists = get_active_site_dists(atoms=atoms, site_indices=site_indices)

    edges_list = get_edges_list_from_connectivity(connectivity=connectivity)
    edge_index = torch.tensor(edges_list, dtype=torch.long, device=device).reshape(2,-1)

    #embed whether a site is active or not
    active_sites = torch.zeros((len(atoms),), dtype=torch.long, device=device)   # 21 atoms
    active_sites[site_indices]+=1
    #Construct the graph
    graph = Graph(
        x=x,
        edge_index=edge_index,
        rate=get_rate_from_atoms(
            atoms=atoms, 
            device=device, 
            use_log=use_log
            ) if "rate" in condition_keys else None,
        e_form=get_stability_measure_from_atoms(
            atoms=atoms, 
            device=device,
            include_recon_label=include_recon_label
            ) if "e_form" in condition_keys else None,
        pos=torch.tensor(atoms.positions, dtype=torch.float, device=device),
        edge_attr=None,
        active_sites=active_sites,
        active_site_dists=torch.tensor(active_site_dists, dtype=torch.float, device=device)
    )
    return graph


def get_rate_from_atoms(
        atoms:Atoms,
        device:str=None,
        use_log:bool=True
    ):
    if "rate" in atoms.info:
        rate = torch.tensor(atoms.info["rate"], device=device, dtype=torch.float)
        if use_log:
            return torch.log10(rate)
        else:
            return rate
    else:
        return None


def get_stability_measure_from_atoms(
        atoms:Atoms,
        device:str=None,
        include_recon_label:bool=False
    ):
    if "e_form" in atoms.info:
        e_form = atoms.info["e_form"]
        if include_recon_label:
            raise Exception("Not implemented yet")
        else:
            return torch.tensor(e_form, device=device, dtype=torch.float)
    else:
        return None


def get_graph_from_datadict(
        datadict:dict, 
        template_atoms:Atoms, 
        element_pool:list, 
        condition_key:str=None,
    ):
    template_atoms.symbols = datadict["elements"]
    graph = get_graph_from_atoms(
        atoms=template_atoms,
        element_pool=element_pool,
        condition_key=None
    )
    if condition_key is not None:
        if condition_key in datadict:
            graph.y = datadict[condition_key]
        else:
            raise Exception(f"condition key {condition_key} is not available in datadict, having: {datadict.keys()}") 
    return graph

def get_dataset_from_datadicts(
        datadicts:list, 
        template_atoms:Atoms, 
        element_pool:list, 
        condition_key:str=None
    ):

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


def get_dataset_from_atoms_list(
        atoms_list:list,
        element_pool:list,
        condition_keys:list=None,
        device:str=None,
        graph_kwargs:dict={}
    ):
    graph_list = [
        get_graph_from_atoms(
            atoms=atoms, 
            element_pool=element_pool, 
            condition_keys=condition_keys,
            device=device,
            **graph_kwargs
        )
        for atoms in atoms_list
    ]
    return GraphDataset(graph_list=graph_list)


def get_dataloaders_from_datadicts(
        data_dicts:list, 
        element_pool:list,
        template_atoms:Atoms,
        batch_size:int=42,
        condition_key:str="class", 
        train_val_split:float=0.1,
        do_initial_shuffling:bool=True,
        loader_kwargs:dict={} 
    ):
    if do_initial_shuffling:
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


def get_dataloaders_from_atoms_list(
        atoms_list:list, 
        element_pool:list,
        batch_size:int=42,
        condition_keys:list=["rate"], 
        train_val_split:float=0.2,
        do_initial_shuffling:bool=True,
        do_train_shuffling:bool=True,
        device:str=None,
        random_seed:int=42,
        loader_kwargs:dict={},
        graph_kwargs:dict={} 
    ):
    if do_initial_shuffling:
        random.seed(random_seed)
        random.shuffle(atoms_list)

    split_index = int((1-train_val_split)*len(atoms_list))
   
    train_dataset = get_dataset_from_atoms_list(
        atoms_list=atoms_list[:split_index],
        element_pool=element_pool,
        condition_keys=condition_keys,
        device=device,
        graph_kwargs=graph_kwargs
    )

    val_dataset = get_dataset_from_atoms_list(
        atoms_list=atoms_list[split_index:],
        element_pool=element_pool,
        condition_keys=condition_keys,
        device=device,
        graph_kwargs=graph_kwargs
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=do_train_shuffling,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )
    return train_loader, val_loader


def get_train_val_atoms_list(
        atoms_list:list,
        train_val_split:float=0.2,
        do_initial_shuffling:bool=True,
        random_seed:int=42,
    ):
    if do_initial_shuffling:
        random.seed(random_seed)
        random.shuffle(atoms_list)
    split_index = int((1-train_val_split)*len(atoms_list))
    return atoms_list[:split_index], atoms_list[split_index:]
    

def get_train_val_dataloaders(
        train_atoms_list:list,
        val_atoms_list:list,
        element_pool:list,
        condition_keys:list=["rate"],
        batch_size:int=42,
        graph_kwargs:dict={},
        loader_kwargs:dict={},
        do_shuffling:bool=True,
        device:str=None,
    ):

    train_dataset = get_dataset_from_atoms_list(
        atoms_list=train_atoms_list,
        element_pool=element_pool,
        condition_keys=condition_keys,
        device=device,
        graph_kwargs=graph_kwargs
    )

    val_dataset = get_dataset_from_atoms_list(
        atoms_list=val_atoms_list,
        element_pool=element_pool,
        condition_keys=condition_keys,
        device=device,
        graph_kwargs=graph_kwargs
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=do_shuffling,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=do_shuffling,
        **loader_kwargs
    )
    return train_loader, val_loader


