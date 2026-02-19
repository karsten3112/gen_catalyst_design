from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax as att_softmax
from pytorch_lightning import LightningModule

import torch.nn as nn
import torch

class SingleMessageLayer(MessagePassing):
    def __init__(
            self,
            input_dim:int,
            output_dim:int,
            message_dim:int=8,
            activation_func=nn.ReLU(),
            aggr = 'mean'

        ):
        super().__init__(aggr)

        self.psi_network = nn.Sequential(
            nn.Linear(2*input_dim, message_dim),
            activation_func,
            nn.Linear(message_dim, message_dim),
            activation_func,
            nn.Linear(message_dim, message_dim)
        )
        self.phi_network = nn.Sequential(
            nn.Linear(message_dim+input_dim, message_dim+output_dim),
            activation_func,
            nn.Linear(message_dim+output_dim, message_dim+output_dim),
            activation_func,
            nn.Linear(message_dim+output_dim, output_dim),
        )

        self.const_state_dict = {
            "input_dim":input_dim,
            "output_dim":output_dim,
            "message_dim":message_dim,
        }

    def forward(self, x_t, edge_index):
        #print(edge_index.shape)
        aggregated_messages = self.propagate(edge_index=edge_index, x=x_t)
        #we let the time embedding work on the global aggregation
        x_t = self.phi_network(torch.hstack([x_t, aggregated_messages])) #time_embedded
        return x_t
        
    
    def message(self, x_i, x_j, index):
        concatenated_x = torch.hstack([x_i, x_j])
        message = self.psi_network(concatenated_x)
        return message


class rateGNN(MessagePassing):
    def __init__(
            self,
            element_pool:list=["Rh", "Cu", "Au", "Pd"], 
            n_elements:int=21,
            message_dim:int=8,
            n_hidden_layers:int=1,
            hidden_dim_rep:int=8,
            aggr = 'sum',
            device=None,
        ):
        super().__init__(aggr)
        self.n_elements = n_elements
        self.hidden_dim_rep = hidden_dim_rep
        self.n_hidden_layers = n_hidden_layers
        self.message_dim = message_dim
        self.aggr = aggr
        self.device = device
        self.element_pool = element_pool
        self.active_site_embedding = nn.Embedding(num_embeddings=2, embedding_dim=message_dim)
        self.element_embeddings = nn.Embedding(num_embeddings=len(element_pool), embedding_dim=message_dim)
        self.site_embeddings = nn.Embedding(num_embeddings=21, embedding_dim=message_dim)
        self.input_layer = SingleMessageLayer(
            input_dim=message_dim,#input_dim,
            output_dim=message_dim,
            message_dim=message_dim,
            aggr=aggr
        )
        hidden_layers = [
            SingleMessageLayer(
                input_dim=message_dim,
                message_dim=message_dim, 
                output_dim=hidden_dim_rep,
                aggr=aggr
            ) for _ in range(n_hidden_layers)
        ]
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.rate_mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim_rep, out_features=hidden_dim_rep),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_rep, out_features=hidden_dim_rep),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_rep, out_features=1)
        )
        self.pool = global_add_pool

    @property
    def const_state_dict(self):
        state_dict = {
            "network_type":"GNN",
            "element_pool":self.element_pool, 
            "n_elements":self.n_elements,
            "message_dim":self.message_dim,
            "n_hidden_layers":self.n_hidden_layers,
            "hidden_dim_rep":self.hidden_dim_rep
        }
        return state_dict

    def representation(self, x_t, batch):
        edge_index = batch.edge_index
        x_t = x_t @ self.element_embeddings.weight
        
        #ids = torch.arange(self.n_elements, device=self.device).repeat(batch.num_graphs)
        #embedded_sites = self.site_embeddings(ids)#self.active_site_embedding(ids)
        x_t = self.input_layer.forward(
            x_t,#torch.hstack([x_t, embedded_sites]), 
            edge_index=edge_index
        )
        for message_passing_layer in self.hidden_layers:
            x_t = message_passing_layer.forward(
                x_t, 
                edge_index=edge_index
            )
        return x_t

    def forward(self, x_t, batch):
        x_t = self.representation(x_t=x_t, batch=batch)
        x_t_pooled = self.pool(x_t, batch.batch)
        reaction_rate = self.rate_mlp(x_t_pooled)
        return reaction_rate


class ReactionRateModule(LightningModule):
    def __init__(self, reaction_rate_nn:rateGNN=None, lr:float=1e-4, weight_decay:float=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.reaction_rate_nn = reaction_rate_nn
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        reaction_rate = self.rate_prediction(x_t=batch.x*1.0, batch=batch)
        mse_loss = self.mse_loss(reaction_rate.squeeze(), batch.y)
        self.log("train_loss", mse_loss, on_epoch=True, batch_size=batch.batch_size)
        return mse_loss
    
    def rate_prediction(self, x_t, batch):
        reaction_rate = self.reaction_rate_nn(x_t, batch)
        return reaction_rate
    
    def get_nn_from_checkpoint(self, nn_params):
        nn_type = nn_params.pop("network_type")
        if nn_type == "GNN":
            reac_rate_nn = rateGNN(**nn_params)
        else:
            raise Exception("Network has not been implemented")
        return reac_rate_nn
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["nn_rate_params"] = self.reaction_rate_nn.const_state_dict
        return super().on_save_checkpoint(checkpoint)
    
    def on_load_checkpoint(self, checkpoint):
        nn_rate_params = checkpoint["nn_rate_params"]
        self.reaction_rate_nn = self.get_nn_from_checkpoint(nn_params=nn_rate_params)
        return super().on_load_checkpoint(checkpoint)
