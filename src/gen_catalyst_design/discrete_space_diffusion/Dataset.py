from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
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
            y = None,
            active_sites = None, 
            pos = None
        ):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.active_sites = active_sites
    
    def to_elems(self, element_pool:list):
        indices = torch.argmax(self.x, dim=-1)
        return [element_pool[index] for index in indices]

    def to_atoms(self, element_pool:list):
        elements = self.to_elems(element_pool=element_pool)
        updated_elements = ["O" if elem == "(X)" else elem for elem in elements]
        atom_list = []
        for element, position in zip(updated_elements, self.pos):
            atom = Atom(symbol=element, position=position.numpy())
            atom_list.append(atom)
        return Atoms(atom_list)    

    def update_x_from_elements(self, elements:list):
        x = embed_elements_as_onehot(elements=elements, element_pool=self.element_pool)
        self.x = x


class GraphDataset(Dataset):
    def __init__(self, graph_list, transform = None):
        super().__init__(transform)
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
    return [element_pool[index] for index in indices]

def embed_elements_as_onehot(elements:list, element_pool:list):
    mapping_dict = {element:i for i, element in enumerate(element_pool)}
    return torch.stack([get_onehot(element=element, mapping_dict=mapping_dict) for element in elements])


def embed_cluster_as_onehots(atoms:Atoms, element_pool:list):
    elements = atoms.get_chemical_symbols()
    return embed_elements_as_onehot(elements=elements, element_pool=element_pool)


def get_onehot(element:str, mapping_dict:dict):
    onehot = F.one_hot(torch.tensor(mapping_dict[element]), len(mapping_dict))
    return onehot

def get_graph_from_atoms(
        atoms:Atoms,
        element_pool:list,
        condition_key:str=None,
        mark_active_sites:bool=True,
        use_edge_attr:bool=True
    ):

    #Get active site embedding - maybe change to index instead and use nn.Embedding
    active_sites = None
    if mark_active_sites or use_edge_attr:
        active_sites = torch.zeros(size=(len(atoms), 2))
        indices_site = atoms.info["indices_site"]
        for i in range(len(atoms)):
            if i in indices_site:
                active_sites[i,0] += 1
            else:
                active_sites[i,1] += 1

    elements = atoms.get_chemical_symbols()
    x = embed_elements_as_onehot(elements=elements, element_pool=element_pool)
    edges_list = get_edges_list_from_connectivity(atoms.info["connectivity"])
    edge_index = torch.tensor(edges_list, dtype=torch.long).reshape(2,-1)

    #Get edge attributes as sum of one-hots from elements + active-site onehot - maybe change to index and use nn.Embedding
    edge_attr = None
    if use_edge_attr:
        x_stacked = torch.hstack([x, active_sites])
        edge_attr = x_stacked[edge_index[0]] + x_stacked[edge_index[1]]

    #Construct the graph
    graph = Graph(
        x=x,
        edge_index=edge_index,
        pos=torch.tensor(atoms.positions),
        active_sites=active_sites,
        edge_attr=edge_attr
    )

    #Assign condition to graph
    if condition_key is not None:
        if condition_key in atoms.info:
            graph.y = atoms.info[condition_key]
        else:
            raise Exception(f"condition key {condition_key} is not available in datadict, having: {atoms.info.keys()}")
    return graph

def get_graph_from_datadict(
        datadict:dict, 
        template_atoms:Atoms, 
        element_pool:list, 
        condition_key:str=None
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
        condition_key:str=None,
        graph_kwargs:dict={}
    ):
    graph_list = [
        get_graph_from_atoms(
            atoms=atoms, 
            element_pool=element_pool, 
            condition_key=condition_key,
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
        condition_key:str="class", 
        train_val_split:float=0.1,
        do_initial_shuffling:bool=True,
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
        condition_key=condition_key,
        graph_kwargs=graph_kwargs
    )
    val_dataset = get_dataset_from_atoms_list(
        atoms_list=atoms_list[split_index:],
        element_pool=element_pool,
        condition_key=condition_key,
        graph_kwargs=graph_kwargs
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

