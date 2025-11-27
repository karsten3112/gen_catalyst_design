import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Dataset, InMemoryDataset



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



class OneHotDataset(Dataset):
    def __init__(self, xs:torch.tensor=None, conds:torch.tensor=None):
        super().__init__()
        self.xs = xs
        self.conds = conds

    def __getitem__(self, index):
        return (self.xs[index], self.conds[index])
    
    def __len__(self):
        return len(self.xs)

    def get_xs_from_atoms_list(self, atoms_list:list, element_pool:list):
       xs = [self.get_x_from_elems(elems_list=atoms.get_chemical_symbols(), element_pool=element_pool) for atoms in atoms_list]
       return torch.stack(xs)
    
    def get_x_from_elems(self, elems_list:list, element_pool:list):
        mapping_dict = {elem:i for i, elem in enumerate(element_pool)}
        x = torch.stack([F.one_hot(torch.tensor(mapping_dict[elem]), num_classes=len(element_pool)) for elem in elems_list])
        return x
    
    def get_conditional_vect(self, data_dict:dict, conditionals:list):
        for conditional in conditionals:
            pass

    def get_xs_conds_from_data_dicts(self, data_dicts:list, element_pool:list, conditionals:torch.tensor=None):
        xs = []
        conds = []
        for data_dict in data_dicts:
            x = self.get_x_from_elems(elems_list=data_dict["elements"], element_pool=element_pool)
            xs.append(x)
        return torch.stack(xs), conditionals
    
    def add_data_from_data_dicts(self, data_dicts:list, element_pool:list, conditionals:list=None):
        self.xs, self.conds = self.get_xs_conds_from_data_dicts(data_dicts=data_dicts, element_pool=element_pool, conditionals=conditionals)
        