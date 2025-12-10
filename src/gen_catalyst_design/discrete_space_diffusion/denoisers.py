import torch
import torch.nn as nn
import torch.nn.functional as F
from .conditioning import ConditioningEmbedder
from .schedulers import DiscreteTimeScheduler
from .Dataset import GraphDataset, Graph
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from ase_ml_models.pyg import get_edges_list_from_connectivity

# -------------------------------------------------------------------------------------
# DISCRETE SPACE DENOISER BASE-CLASS
# -------------------------------------------------------------------------------------

class DiscreteSpaceDenoiser(nn.Module):
    def __init__(
            self, 
            element_pool:list,
            cond_embedder:ConditioningEmbedder
        ):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.element_pool = element_pool
        if "(X)" in element_pool:
            self.absorbing_state = True
            self.absorbing_state_index = self.get_absorbing_state_index(element_pool=element_pool)
        else:
            self.absorbing_state = False
            self.absorbing_state_index = None

    def get_absorbing_state_index(self, element_pool:list):
        for i, element in enumerate(element_pool):
            if element == "(X)":
                return i

    def forward(self, x_t:torch.tensor, batch, time, scheduler:DiscreteTimeScheduler, drop_condition:bool, **kwargs):
        raise Exception("must be implemented by sub-class")
    
    def get_sample_loader(self, dataset, batch_size, shuffle:bool=True):
        raise Exception("must be implemented by sub-class")

    def get_time_embedding(self, time, t_init, t_final):
        cos = torch.cos(torch.pi/(t_final-t_init)*time).view(-1,1)
        sin = torch.sin(torch.pi/(t_final-t_init)*time).view(-1,1)
        return torch.hstack([cos, sin])
    
    def get_probabilities_from_logits(self, logits):
        probabilities = F.log_softmax(logits, dim=-1) 
        return torch.exp(probabilities)
    
    def get_guided_logits(self, batch, time, scheduler:DiscreteTimeScheduler, guidance_scale:float=2.0, **kwargs):
        logits = [
            self.forward(
            x_t=batch.x*1.0, 
            batch=batch, 
            time=time, 
            scheduler=scheduler, 
            drop_condition=drop_cond) for drop_cond in [True, False]
        ]
        logits_guided = logits[0] + guidance_scale*(logits[1] - logits[0])
        return logits_guided
    
    def get_logits(self, x_t, batch, time, scheduler:DiscreteTimeScheduler, drop_cond:bool):
        logits = self.forward(
            x_t=x_t, 
            batch=batch, 
            time=time, 
            scheduler=scheduler, 
            drop_condition=drop_cond
        )
        return logits
        

    def denoise_batch(self, batch, scheduler:DiscreteTimeScheduler, guidance_scale:float=2.0, timesteps:torch.tensor=None, log_all_timesteps:bool=False):
        batch_list = []
        if timesteps is None:
            timesteps = torch.arange(scheduler.t_init, scheduler.t_final, 1).flip(dims=(0,))
        for timestep in timesteps:
            ts = timestep*torch.ones(size=(batch.batch_size,))
            xs_denoised = self.single_denoise_step(batch=batch, time=ts, scheduler=scheduler, guidance_scale=guidance_scale)
            if log_all_timesteps:
                batch_list.append(xs_denoised)
            else:
                batch_list = [xs_denoised]
        return batch_list

    def get_distribution(self, probabilites:torch.tensor) -> Categorical:
        return Categorical(probs=probabilites)

    def sample_onehot_vectors(self, probabilities:torch.tensor):
        distribution = self.get_distribution(probabilites=probabilities)
        samples = distribution.sample()
        onehots = F.one_hot(samples, num_classes=len(self.element_pool))
        return onehots

    def single_denoise_step(self, batch, time, scheduler:DiscreteTimeScheduler, guidance_scale:float=2.0):
        probs = self.get_transition_probabilities(batch=batch, time=time, scheduler=scheduler, guidance_scale=guidance_scale)
        xs_denoised = self.sample_onehot_vectors(probabilities=probs)
        batch.x = xs_denoised
        return xs_denoised

    def get_sample_dataset(self, noised_xs, conditionings, template_atoms):
        raise Exception("must be implemented by sub-class")

    @property
    def const_state_dict(self):
        state_dict = {"condition_info":self.cond_embedder.const_state_dict}
        return state_dict
    
# -------------------------------------------------------------------------------------
# GNN-DENOISER CLASSES
# -------------------------------------------------------------------------------------


class SingleMessageLayer(MessagePassing):
    def __init__(
            self,
            input_dim:int,
            output_dim:int,
            message_dim:int=8, 
            conditioning_dim:int=8,
            activation_func=torch.nn.ReLU(),
            time_embedding_dim:int=2, 
            aggr = 'sum'
        ):
        super().__init__(aggr)

        self.psi_network = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            activation_func,
            nn.Linear(message_dim, message_dim)
        )
        self.phi_network = nn.Sequential(
            nn.Linear(message_dim+input_dim+time_embedding_dim, message_dim+output_dim),
            activation_func,
            nn.Linear(message_dim+output_dim, output_dim),
        )
        self.gamma_net = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            activation_func,
            nn.Linear(conditioning_dim, output_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            activation_func,
            nn.Linear(conditioning_dim, output_dim)
        )

        self.const_state_dict = {
            "input_dim":input_dim,
            "output_dim":output_dim,
            "message_dim":message_dim,
            "time_embedding_dim":time_embedding_dim              
        }

    def forward(self, x_t, edge_index, conds_embedded, time_embedded):
        aggregated_messages = self.propagate(edge_index=edge_index, x=x_t)
        #we let the time embedding work on the global aggregation
        x_t = self.phi_network(torch.hstack([x_t, aggregated_messages, time_embedded]))
        #We shift the final representation using gamma, and beta MLP's
        gamma, beta = self.gamma_net(conds_embedded), self.beta_net(conds_embedded)
        return gamma*x_t + beta
    
    def message(self, x_j):
        return self.psi_network(x_j)
    

class DiscreteGNNDenoiser(DiscreteSpaceDenoiser):
    def __init__(self, 
                 element_pool:list, 
                 cond_embedder:ConditioningEmbedder,
                 message_dim:int=8,
                 n_hidden_layers:int=1,
                 hidden_dim_rep:int=8,
                 ):
        super().__init__(element_pool, cond_embedder)
        input_layer = SingleMessageLayer(
            input_dim=len(element_pool),
            output_dim=message_dim,
            message_dim=message_dim, 
            conditioning_dim=self.cond_embedder.embedding_dim
        )
        output_layer = SingleMessageLayer(
            input_dim=hidden_dim_rep,
            message_dim=message_dim, 
            output_dim=len(self.element_pool), 
            conditioning_dim=self.cond_embedder.embedding_dim
        )
        hidden_layers = [
            SingleMessageLayer(
                input_dim=message_dim,
                message_dim=message_dim, 
                output_dim=hidden_dim_rep, 
                conditioning_dim=self.cond_embedder.embedding_dim
                ) for _ in range(n_hidden_layers)]
        
        self.message_passing_layers = nn.ModuleList([input_layer] + hidden_layers + [output_layer])
        self.hidden_dim_rep = hidden_dim_rep
        self.n_hidden_layers = n_hidden_layers
        self.message_dim = message_dim
    
    def forward(self, x_t, batch, time, scheduler:DiscreteTimeScheduler, drop_condition:bool):
        edge_index, conds, batch_indices = batch.edge_index, batch.y, batch.batch
        time_embedded = self.get_time_embedding(time=time[batch_indices], t_init=scheduler.t_init, t_final=scheduler.t_final)
        embedded_conds = self.cond_embedder.forward(condition=conds[batch_indices], drop_condition=drop_condition)
        for message_passing_layer in self.message_passing_layers:
            x_t = message_passing_layer.forward(x_t, edge_index=edge_index, conds_embedded=embedded_conds, time_embedded=time_embedded)
        return x_t

    def get_sample_loader(self, dataset, batch_size, shuffle:bool=True):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def get_sample_dataset(self, noised_xs, conditionings, template_atoms):
        edges_list = get_edges_list_from_connectivity(template_atoms.info["connectivity"])
        edge_index = torch.tensor(edges_list, dtype=torch.long).reshape(2,-1)
        graphs = [
            Graph(x=noised_x, edge_index=edge_index, y=conditioning, pos=torch.tensor(template_atoms.positions))
            for noised_x, conditioning in zip(noised_xs, conditionings)
        ]
        return GraphDataset(graph_list=graphs)

    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        denoiser_info = {
            "denoiser_type":"DiscreteGNNDenoiser",
            "message_dim":self.message_dim,
            #"layers_info":[message_layer.const_state_dict for message_layer in self.message_passing_layers],
            "n_hidden_layers":self.n_hidden_layers,
            "hidden_dim_rep":self.hidden_dim_rep
        }
        state_dict.update(denoiser_info)
        return state_dict

