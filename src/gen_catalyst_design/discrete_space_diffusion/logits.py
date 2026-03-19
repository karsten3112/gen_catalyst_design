import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention, global_add_pool, global_mean_pool
from .conditioning import NoneConditioning
from .Dataset import get_dataset_from_atoms_list
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax


# -------------------------------------------------------------------------------------
# LOGIT PREDICTOR BASE CLASS
# -------------------------------------------------------------------------------------

class LogitPredictor(nn.Module):
    def __init__(
            self,
            num_elements:int,
            hidden_rep_dim:int=32,
            conditioning_dim:int=None,
            activation_func:callable=nn.ReLU(),
            device=None
        ):
        super().__init__()
        self.device = device
        self.num_elements = num_elements
        self.hidden_rep_dim = hidden_rep_dim
        if conditioning_dim is None:
            self.conditioning_dim = NoneConditioning().embedding_dim
        else:
            self.conditioning_dim = conditioning_dim

        #head for predicting rates
        self.rate_head = nn.Sequential(
            nn.Linear(in_features=hidden_rep_dim, out_features=hidden_rep_dim),
            activation_func,
            nn.Linear(in_features=hidden_rep_dim, out_features=hidden_rep_dim),
            activation_func,
            nn.Linear(in_features=hidden_rep_dim, out_features=1)
        )

        #head for predicting logits
        self.logit_head = nn.Sequential(
            nn.Linear(in_features=hidden_rep_dim + conditioning_dim, out_features=hidden_rep_dim),
            activation_func,
            nn.Linear(in_features=hidden_rep_dim, out_features=hidden_rep_dim),
            activation_func,
            nn.Linear(in_features=hidden_rep_dim, out_features=num_elements)
        )

    def set_device(self, device):
        self.device = device

    def get_state_dict(self):
        state_dict = {
            "num_elements":self.num_elements,
            "conditioning_dim":self.conditioning_dim,
            "hidden_rep_dim":self.hidden_rep_dim
        }
        return state_dict

    def forward(self, batch, time, embedded_condition):
        raise NotImplementedError("logit estimation must be inferred by sub-class")

    def get_x0_rate_prediction(self, batch, time, embedded_condition):
        x_t = self.forward(
            batch=batch,
            time=time,
            embedded_condition=embedded_condition
        )
        ctx = global_mean_pool(x_t, batch.batch)
        return self.rate_head(ctx)

    def get_logits(self, batch, time, embedded_condition):
        x_t = self.forward(
            batch=batch,
            time=time,
            embedded_condition=embedded_condition
        )
        if embedded_condition is not None:
            return self.logit_head(torch.hstack([x_t, embedded_condition]))
        else:
            return self.logit_head(x_t)

    def get_sample_loader(self, atoms_list:list, element_pool:list, batch_size:int, condition_keys:list=None, dataset_kwargs:dict={}):
        raise NotImplementedError("sample loader must be implemented by sub-class")

    def get_probs_from_logits(self, logits):
        return F.softmax(logits, dim=-1)


class FiLMNet(nn.Module):
    def __init__(
            self,
            conditining_dim:int,
            output_dim:int,
            embedding_dim:int,
            activation_func:callable=nn.ReLU()
        ):
        super().__init__()
    
        self.gamma = nn.Sequential(
            nn.Linear(conditining_dim+embedding_dim, output_dim),
            activation_func,
            nn.Linear(output_dim, output_dim),
            activation_func,
            nn.Linear(output_dim, output_dim),
        )

        self.beta = nn.Sequential(
            nn.Linear(conditining_dim+embedding_dim, output_dim),
            activation_func,
            nn.Linear(output_dim, output_dim),
            activation_func,
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, message, embedded_condition):
        features = torch.hstack([message, embedded_condition])
        return self.gamma(features), self.beta(features)

    def modulate_representation(self, message, embeddded_condition):
        #if embeddded_condition is None:
        #    gamma, beta = 1.0, 0.0
        #else:
        gamma, beta = self.forward(
            message=message,
            embedded_condition=embeddded_condition
        )
        return (1.0+gamma)*message + beta

class ContentDistanceAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, rbf_dim: int = 32, n_heads: int = 4):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Distance -> attention bias (your existing idea)
        self.centers = nn.Parameter(torch.linspace(0.0, 12.0, rbf_dim))
        self.width = nn.Parameter(torch.tensor(1.0))
        self.edge_mlp = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)   # scalar bias per (i,j)
        )

    def rbf(self, d):
        diff = d[..., None] - self.centers[None, ...]
        return torch.exp(-(diff**2) / (self.width.abs() + 1e-6))

    def forward(self, h, pos, batch):
        H, mask = to_dense_batch(h, batch)     # [B, M, D]
        P, _    = to_dense_batch(pos, batch)   # [B, M, 3]
        B, M, D = H.shape

        d = torch.cdist(P, P)                  # [B, M, M]
        rbf = self.rbf(d)                      # [B, M, M, rbf_dim]
        dist_bias = self.edge_mlp(rbf).squeeze(-1)  # [B, M, M]

        # Q,K,V: [B, M, H, d_head]
        Q = self.q_proj(H).view(B, M, self.n_heads, self.d_head)
        K = self.k_proj(H).view(B, M, self.n_heads, self.d_head)
        V = self.v_proj(H).view(B, M, self.n_heads, self.d_head)

        # attention logits: [B, H, M, M]
        attn_logits = torch.einsum("bmhd,bnhd->bhmn", Q, K) / (self.d_head ** 0.5)
        attn_logits = attn_logits + dist_bias[:, None, :, :]  # add distance bias to all heads

        # masks
        key_mask = mask[:, None, None, :].expand(B, self.n_heads, M, M)  # mask keys
        attn_logits = attn_logits.masked_fill(~key_mask, -1e9)

        # optional: exclude self
        eye = torch.eye(M, device=h.device, dtype=torch.bool)[None, None, :, :]
        attn_logits = attn_logits.masked_fill(eye, -1e9)

        attn = F.softmax(attn_logits, dim=-1)  # [B, H, M, M]
        
        # ctx: [B, H, M, d_head] -> [B, M, D]
        ctx = torch.einsum("bhmn,bnhd->bmhd", attn, V).reshape(B, M, D)
        ctx = self.out_proj(ctx)

        # back to sparse: [N, D]
        return ctx[mask]


class EmbeddingBlock(nn.Module):
    def __init__(
            self, 
            num_elements:int,
            embedding_dim:int=64,
            time_embedding_dim:int=32, 
            activation_func:callable=nn.ReLU(),
            device=None
        ):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.device = device
        self.embedding_dim = embedding_dim
        #Element embedddings
        self.element_embeddings = nn.Embedding(
            num_embeddings=num_elements, 
            embedding_dim=embedding_dim
        )

        self.element_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        #Embedding of active sites
        self.active_site_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embedding_dim
        )

        #Embedder of distance to active_sites
        self.dist_embedder = nn.Sequential(
            nn.Linear(4, embedding_dim),
            activation_func,
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.geom_rep_proj = nn.Linear(2*embedding_dim, embedding_dim, bias=False)

        #Embedder of time
        self.time_nn_embedder = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            activation_func,
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

    def set_device(self, device):
        self.device = device

    def get_time_embedding(self, time):
        indices = torch.arange(0, self.time_embedding_dim, 1, device=self.device)
        angle_rates = 1 / torch.pow(10000, (2 * (indices // 2)) / self.time_embedding_dim)
        timesteps=time[:,None]*angle_rates[None,:]
        time_embedded = torch.zeros_like(timesteps, device=self.device)
        time_embedded[:, 0::2] = torch.sin(timesteps[:, 0::2])
        time_embedded[:, 1::2] = torch.cos(timesteps[:, 1::2])
        return self.time_nn_embedder(time_embedded)

    def forward(self, batch, time):
        x_t, batch_indices, active_sites, active_site_dists = (
            batch.x,
            batch.batch, 
            batch.active_sites,
            batch.active_site_dists
        )
        #Embed the atomic species via learnable embeddings
        indices = torch.argmax(x_t, dim=1)
        x_t = self.element_embeddings(indices)

        #Embed geometrical information via active site embeddings
        active_site_emb = self.active_site_embedding(active_sites)
        #Embed geometrical information via distance to active sites
        active_site_dist_emb = self.dist_embedder(active_site_dists)

        #Concatenate geometrical information
        geom_rep = torch.hstack([active_site_emb, active_site_dist_emb])

        #Embed time by using sinusoidal embeddings
        time_embedded = self.get_time_embedding(time=time)[batch_indices]
        return x_t, geom_rep, time_embedded

    def project_embeddings(self, x_t, geom_rep):
        return self.element_proj(x_t) + self.geom_rep_proj(geom_rep)


class MessagePassingBlock(MessagePassing):
    def __init__(
            self,
            input_dim:int=32,
            output_dim:int=32,
            message_dim:int=32,
            conditioning_dim:int=32,
            time_embedding_dim:int=32,
            activation_func=nn.ReLU(),
            aggr = 'sum'
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
            nn.Linear(message_dim+input_dim+time_embedding_dim, message_dim+output_dim),
            activation_func,
            nn.Linear(message_dim+output_dim, message_dim+output_dim),
            activation_func,
            nn.Linear(message_dim+output_dim, output_dim),
        )

        self.film_net = FiLMNet(
            conditining_dim=conditioning_dim,
            output_dim=message_dim,
            embedding_dim=message_dim
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.const_state_dict = {
            "input_dim":input_dim,
            "output_dim":output_dim,
            "message_dim":message_dim,
            "time_embedding_dim":time_embedding_dim
        }
    
    def forward(self, x_t, edge_index, conds_embedded, time_embedded):
        aggregated_messages = self.propagate(
            edge_index=edge_index, 
            x=x_t,
            conds_embedded=conds_embedded
        )
        #we let the time embedding work on the global aggregation
        x_t_updated = self.phi_network(torch.hstack([x_t, aggregated_messages, time_embedded]))
        return self.layer_norm(x_t + x_t_updated)
    
    def message(self, x_i, x_j, conds_embedded_i):
        message = self.psi_network(torch.hstack([x_i, x_j]))
        modulated_message = self.film_net.modulate_representation(
            message=message,
            embeddded_condition=conds_embedded_i
        )
        return modulated_message

class MPNNLogitPredictor(LogitPredictor):
    def __init__(
            self, 
            num_elements,
            embedding_dim:int=32, 
            hidden_rep_dim = 32, 
            time_embedding_dim = 32,
            conditioning_dim = None,
            n_interaction_blocks:int=3,
            message_dim:int=32, 
            activation_func = nn.ReLU(),
            aggr:str="mean", 
            device=None
        ):
        super().__init__(num_elements, hidden_rep_dim, conditioning_dim, activation_func, device)
        self.n_interaction_blocks = n_interaction_blocks
        self.time_embedding_dim = time_embedding_dim
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim
        self.aggr = aggr

        self.embedding_block = EmbeddingBlock(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
            time_embedding_dim=time_embedding_dim,
            activation_func=activation_func,
            device=device
        )        

        self.interaction_blocks = nn.ModuleDict(
            {f"block_{i}": 
             MessagePassingBlock(
                input_dim=embedding_dim if i == 0 else hidden_rep_dim,
                output_dim=hidden_rep_dim,
                message_dim=message_dim,
                time_embedding_dim=time_embedding_dim, 
                conditioning_dim=conditioning_dim,
                activation_func=activation_func,
                aggr=aggr,
            )
            for i in range(n_interaction_blocks)
            }
        )

        self.distance_pooling = ContentDistanceAttentionPooling(
            hidden_dim=hidden_rep_dim,
            rbf_dim=message_dim,
            n_heads=4
        )

        self.logit_head = nn.Sequential(
            nn.Linear(in_features=2*hidden_rep_dim + conditioning_dim, out_features=hidden_rep_dim),
            activation_func,
            nn.Linear(in_features=hidden_rep_dim, out_features=hidden_rep_dim),
            activation_func,
            nn.Linear(in_features=hidden_rep_dim, out_features=num_elements)
        )

    def set_device(self, device):
        super().set_device(device)
        self.embedding_block.set_device(device=device)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        extra_info = {
            "logit_predictor_type":"MPNNLogitPredictor",
            "time_embedding_dim":self.time_embedding_dim,
            "embedding_dim":self.embedding_dim,
            "n_interaction_blocks":self.n_interaction_blocks,
            "message_dim":self.message_dim,
        }
        state_dict.update(extra_info)
        return state_dict

    def forward(self, batch, time, embedded_condition):
        x_t, geom_rep, time_embedded = self.embedding_block(batch=batch, time=time)
        x_t = self.embedding_block.project_embeddings(x_t=x_t, geom_rep=geom_rep)
        for block_id in self.interaction_blocks:
            x_t = self.interaction_blocks[block_id](
                x_t=x_t, 
                edge_index=batch.edge_index, 
                conds_embedded=embedded_condition, 
                time_embedded=time_embedded
            )
        return x_t
    
    def get_logits(self, batch, time, embedded_condition):
        x_t = self.forward(
            batch=batch,
            time=time,
            embedded_condition=embedded_condition
        )
        ctx = self.distance_pooling(h=x_t, batch=batch.batch, pos=batch.pos)
        if embedded_condition is not None:
            return self.logit_head(torch.hstack([x_t, ctx, embedded_condition]))
        else:
            return self.logit_head(torch.hstack([x_t, ctx]))
    
    def get_sample_loader(self, atoms_list, element_pool, batch_size, condition_keys = None, dataset_kwargs = {}):
        from torch_geometric.loader import DataLoader
        sample_data = get_dataset_from_atoms_list(
            atoms_list=atoms_list,
            element_pool=element_pool,
            condition_keys=condition_keys,
            device=self.device,
            **dataset_kwargs
        )
        return DataLoader(dataset=sample_data, batch_size=batch_size, shuffle=False)
    

class TransformerBlock(MessagePassingBlock):
    def __init__(self, input_dim = 32, output_dim = 32, message_dim = 32, conditioning_dim = 32, time_embedding_dim = 32, n_heads:int=4, activation_func=nn.ReLU(), aggr='sum'):
        super().__init__(input_dim, output_dim, message_dim, conditioning_dim, time_embedding_dim, activation_func, aggr)
    
        self.distance_pooling = ContentDistanceAttentionPooling(
            hidden_dim=input_dim,
            rbf_dim=message_dim,
            n_heads=n_heads
        )

        self.phi_network = nn.Sequential(
            nn.Linear(input_dim+time_embedding_dim, output_dim),
            activation_func,
            nn.Linear(output_dim, output_dim),
            activation_func,
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x_t, pos, edge_index, conds_embedded, time_embedded, batch_indices):
        x_t_new = self.distance_pooling(h=x_t, batch=batch_indices, pos=pos)
        x_new = self.layer_norm(x_t + x_t_new)
        x_t_updated = self.phi_network(torch.hstack([x_new, time_embedded]))
        return x_t_updated


class TransformerLogitPredictor(LogitPredictor):
    def __init__(
            self, 
            num_elements,
            embedding_dim:int=32, 
            hidden_rep_dim = 32, 
            conditioning_dim = 32,
            time_embedding_dim = 32,
            n_interaction_blocks:int=3,
            n_heads:int=4,
            message_dim:int=32, 
            activation_func = nn.ReLU(),
            aggr:str="sum", 
            device=None
        ):
        super().__init__(num_elements, hidden_rep_dim, conditioning_dim, activation_func, device)
        self.n_interaction_blocks = n_interaction_blocks
        self.time_embedding_dim = time_embedding_dim
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim
        self.n_heads = n_heads
        self.aggr = aggr
        
        self.embedding_block = EmbeddingBlock(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
            time_embedding_dim=time_embedding_dim,
            activation_func=activation_func,
            device=device
        )        

        self.interaction_blocks = nn.ModuleDict(
            {f"block_{i}": 
             TransformerBlock(
                input_dim=embedding_dim if i == 0 else hidden_rep_dim,
                output_dim=hidden_rep_dim,
                message_dim=message_dim,
                time_embedding_dim=time_embedding_dim, 
                conditioning_dim=conditioning_dim,
                activation_func=activation_func,
                n_heads=n_heads,
                aggr=aggr,
            )
            for i in range(n_interaction_blocks)
            }
        )

    def set_device(self, device):
        super().set_device(device)
        self.embedding_block.set_device(device=device)
        #for block in self.interaction_blocks:
        #    self.interaction_blocks[block].head_dim = self.interaction_blocks[block].head_dim.to(device)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        extra_info = {
            "logit_predictor_type":"TransformerLogitPredictor",
            "time_embedding_dim":self.time_embedding_dim,
            "embedding_dim":self.embedding_dim,
            "n_interaction_blocks":self.n_interaction_blocks,
            "message_dim":self.message_dim,
            "n_heads":self.n_heads
        }
        state_dict.update(extra_info)
        return state_dict
    
    def forward(self, batch, time, embedded_condition):
        x_t, geom_rep, time_embedded = self.embedding_block(batch=batch, time=time)
        x_t = self.embedding_block.project_embeddings(x_t=x_t, geom_rep=geom_rep)
        for block_id in self.interaction_blocks:
            x_t = self.interaction_blocks[block_id](
                x_t=x_t,
                pos=batch.pos, 
                edge_index=batch.edge_index, 
                conds_embedded=embedded_condition, 
                time_embedded=time_embedded,
                batch_indices=batch.batch
            )
        return x_t
    
    def get_sample_loader(self, atoms_list:list, element_pool:list, batch_size:int, condition_key:str=None, dataset_kwargs:dict={}):
        from torch_geometric.loader import DataLoader
        sample_data = get_dataset_from_atoms_list(
            atoms_list=atoms_list,
            element_pool=element_pool,
            condition_key=condition_key,
            **dataset_kwargs
        )
        return DataLoader(dataset=sample_data, batch_size=batch_size, shuffle=False)


