from gen_catalyst_design.utils import embed_elements_as_onehot
from torch_geometric.data import Dataset, InMemoryDataset, Data
from ase_ml_models.pyg import get_edges_list_from_connectivity
import torch.nn.functional as F
from ase.atoms import Atoms
from ase.atom import Atom
import torch


class Graph(Data):
    def __init__(
            self,
            x = None, 
            edge_index = None, 
            edge_attr = None, 
            y = None, 
            pos = None
        ):
        super().__init__(x, edge_index, edge_attr, y, pos)
    
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

        