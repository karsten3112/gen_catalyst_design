import torch
import torch.nn as nn
import torch.nn.functional as F
from .conditioning import ConditioningEmbedder
from .schedulers import DiscreteTimeScheduler
from .Dataset import GraphDataset, Graph, get_dataset_from_atoms_list
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing, GlobalAttention, global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as att_softmax
from ase_ml_models.pyg import get_edges_list_from_connectivity
from torch_geometric.utils import to_dense_batch

# -------------------------------------------------------------------------------------
# DISCRETE SPACE DENOISER BASE-CLASS
# -------------------------------------------------------------------------------------

class DiscreteSpaceDenoiser(nn.Module):
    def __init__(
            self, 
            element_pool:list,
            cond_embedder:ConditioningEmbedder=None,
            time_embedding_dim:int=24,
            device=None
        ):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.element_pool = element_pool
        self.time_embedding_dim = time_embedding_dim
        self.device = device
        if "(X)" in element_pool:
            self.absorbing_state = True
            self.absorbing_state_index = self.get_absorbing_state_index(element_pool=element_pool)
        else:
            self.absorbing_state = False
            self.absorbing_state_index = None

    def set_device(self, device):
        self.device = device    

    def get_absorbing_state_index(self, element_pool:list):
        for i, element in enumerate(element_pool):
            if element == "(X)":
                return i

    def forward(self, x_t:torch.tensor, batch, time, drop_condition:bool, **kwargs):
        raise Exception("must be implemented by sub-class")
    
    def get_sample_dataset_from_atoms_list(self, atoms_list:list, dataset_kwargs:dict={}):
        raise Exception("must be implemented by sub-class")

    def get_sample_loader(self, dataset, batch_size, shuffle:bool=True):
        raise Exception("must be implemented by sub-class")
    
    def get_time_embedding(self, time):
        indices = torch.arange(0, self.time_embedding_dim, 1, device=self.device)
        angle_rates = 1 / torch.pow(10000, (2 * (indices // 2)) / self.time_embedding_dim)
        timesteps=time[:,None]*angle_rates[None,:]
        time_embedded = torch.zeros_like(timesteps, device=self.device)
        time_embedded[:, 0::2] = torch.sin(timesteps[:, 0::2])
        time_embedded[:, 1::2] = torch.cos(timesteps[:, 1::2])
        return time_embedded

    def get_probabilities_from_logits(self, logits):
        probabilities = F.log_softmax(logits, dim=-1) 
        return torch.exp(probabilities)
    
    def get_guided_logits(self, batch, time, guidance_scale:float=2.0, **kwargs):
        logits = [
            self.forward(
            x_t=batch.x*1.0, 
            batch=batch, 
            time=time, 
            drop_condition=drop_cond) for drop_cond in [True, False]
        ]
        logits_guided = logits[0] + guidance_scale*(logits[1] - logits[0])
        return logits_guided
    
    def get_logits(self, batch, time, drop_cond:bool):
        logits = self.forward(
            x_t=batch.x*1.0, 
            batch=batch, 
            time=time, 
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
        if self.cond_embedder is None:
            state_dict = {"condition_info":None}
        else:
            state_dict = {"condition_info":self.cond_embedder.const_state_dict}
        return state_dict
    
# -------------------------------------------------------------------------------------
# GNN-DENOISER CLASSES
# -------------------------------------------------------------------------------------

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


class DistanceGlobalPooling(nn.Module):
    def __init__(self, hidden_dim: int, rbf_dim: int = 32):
        super().__init__()
        # Learn a scalar logit from an RBF of distance
        self.centers = nn.Parameter(torch.linspace(0.0, 12.0, rbf_dim))  # adjust max distance
        self.width = nn.Parameter(torch.tensor(1.0))
        self.edge_mlp = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # produces attention logit
        )

    def rbf(self, d):  # d: [...,]
        # Gaussian RBF
        diff = d[..., None] - self.centers[None, ...]
        return torch.exp(-(diff**2) / (self.width.abs() + 1e-6))

    def forward(self, h, pos, batch):
        """
        h:    [N, D]
        pos:  [N, 3]
        batch:[N]
        returns:
          ctx: [N, D] per-node global context
        """
        H, mask = to_dense_batch(h, batch)       # [B, M, D]
        P, _    = to_dense_batch(pos, batch)     # [B, M, 3]
        B, M, D = H.shape

        # Pairwise distances: [B, M, M]
        d = torch.cdist(P, P)  # rotation-invariant

        # Build attention logits from distances: [B, M, M, 1] -> [B, M, M]
        rbf = self.rbf(d)                          # [B, M, M, rbf_dim]
        attn_logits = self.edge_mlp(rbf).squeeze(-1)

        # Mask out padding nodes (keys) and padding queries
        key_mask = mask[:, None, :].expand(B, M, M)   # [B, M, M]
        attn_logits = attn_logits.masked_fill(~key_mask, -1e9)

        # Optional: prevent self from dominating (or allow it)
        eye = torch.eye(M, device=h.device, dtype=torch.bool)[None, :, :]
        attn_logits = attn_logits.masked_fill(eye, -1e9)  # exclude self

        attn = F.softmax(attn_logits, dim=-1)            # [B, M, M]

        # Context: [B, M, D]
        ctx_dense = attn @ H

        # Back to sparse: [N, D]
        ctx = ctx_dense[mask]
        return ctx



class SingleMessageLayer(MessagePassing):
    def __init__(
            self,
            input_dim:int,
            output_dim:int,
            message_dim:int=8,
            conditioning_dim:int=8,
            activation_func=nn.ReLU(),
            time_embedding_dim:int=10, 
            aggr = 'mean',
            non_active_site_scale:float=0.1,

        ):
        super().__init__(aggr)

        self.psi_network = nn.Sequential(
            nn.Linear(6*input_dim+conditioning_dim, 4*message_dim),
            activation_func,
            nn.Linear(4*message_dim, 4*message_dim),
            activation_func,
            nn.Linear(4*message_dim, 2*message_dim),
            activation_func,
            nn.Linear(2*message_dim, message_dim),
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
        self.message_attn_network = nn.Sequential(
            nn.Linear(6*input_dim+conditioning_dim, 3*message_dim),
            activation_func,
            nn.Linear(3*message_dim, 2*message_dim),
            activation_func,
            nn.Linear(2*message_dim, 1),
        )

        self.norm = nn.LayerNorm(output_dim)

        self.non_active_site_scale = non_active_site_scale
        self.const_state_dict = {
            "input_dim":input_dim,
            "output_dim":output_dim,
            "message_dim":message_dim,
            "time_embedding_dim":time_embedding_dim,
            "non_active_site_scale":self.non_active_site_scale              
        }

    def forward(self, x_t, geom_rep, edge_index, conds_embedded, time_embedded):
        aggregated_messages = self.propagate(
            edge_index=edge_index, 
            x=x_t, 
            geom_rep=geom_rep,
            conds_embedded=conds_embedded
        )
        #we let the time embedding work on the global aggregation
        x_t_local = self.phi_network(torch.hstack([x_t, aggregated_messages, time_embedded])) #time_embedded
        h = self.norm(x_t_local)
        #We shift the final representation using gamma, and beta MLP's
        if conds_embedded is None:
            return h
        else:
            gamma, beta = self.gamma_net(conds_embedded), self.beta_net(conds_embedded)
            return gamma*h#+beta#gamma*x_t_updated_rep + beta
    
    def message(self, x_i, x_j, geom_rep_i, geom_rep_j, conds_embedded_i, index):
        if conds_embedded_i is not None:
            concatenated_x = torch.hstack([x_i, x_j, geom_rep_i, geom_rep_j, conds_embedded_i])
        else:
            concatenated_x = torch.hstack([x_i, x_j, geom_rep_i, geom_rep_j])
        attn_score = self.message_attn_network(concatenated_x)
        alpha = att_softmax(attn_score, index)
        message = self.psi_network(concatenated_x)
        return alpha*message
       
    

class DiscreteGNNDenoiser(DiscreteSpaceDenoiser):
    def __init__(
            self,
            element_pool:list, 
            cond_embedder:ConditioningEmbedder=None,
            message_dim:int=8,
            n_hidden_layers:int=1,
            hidden_dim_rep:int=8,
            time_embedding_dim = 10,
            pooling_type:str="att",
            aggr:str="mean"
        ):
        super().__init__(element_pool, cond_embedder, time_embedding_dim)
        input_dim = len(self.element_pool)
        
        self.input_layer = SingleMessageLayer(
            input_dim=message_dim,#*2,#input_dim,
            output_dim=message_dim,
            message_dim=message_dim,
            time_embedding_dim=self.time_embedding_dim, 
            conditioning_dim=self.cond_embedder.embedding_dim if self.cond_embedder is not None else 0,
            aggr=aggr,
        )

        hidden_layers = [
            SingleMessageLayer(
                input_dim=message_dim,
                message_dim=message_dim, 
                output_dim=hidden_dim_rep,
                time_embedding_dim=self.time_embedding_dim,
                conditioning_dim=self.cond_embedder.embedding_dim if self.cond_embedder is not None else 0,
                aggr=aggr
            ) for _ in range(n_hidden_layers)
        ]
        output_dim = hidden_dim_rep
        prob_mlp_in_dim = 2*hidden_dim_rep
        if pooling_type == "att":
            self.pooling = GlobalAttention(
            gate_nn=nn.Linear(output_dim, output_dim),
            nn=nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 1)
                )
            )
            prob_mlp_in_dim*=hidden_dim_rep
        if pooling_type == "mean":
            self.pooling = global_mean_pool
            prob_mlp_in_dim+=hidden_dim_rep
        if pooling_type is None:
            self.pooling = None

        prob_mlp_in_dim += self.cond_embedder.embedding_dim if self.cond_embedder is not None else 0
        self.prob_head = nn.Sequential(
            nn.Linear(in_features=prob_mlp_in_dim, out_features=hidden_dim_rep),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_rep, out_features=hidden_dim_rep),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_rep, out_features=len(element_pool))
        )

        self.rate_head = nn.Sequential(
            nn.Linear(in_features=hidden_dim_rep, out_features=hidden_dim_rep),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_rep, out_features=hidden_dim_rep),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_rep, out_features=1)
        )

        self.dist_embedder = nn.Sequential(
            nn.Linear(4, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )

        #self.dist_pooling = DistanceGlobalPooling(hidden_dim=hidden_dim_rep, rbf_dim=message_dim)
        self.dist_pooling = ContentDistanceAttentionPooling(
            hidden_dim=hidden_dim_rep,
            rbf_dim=message_dim,
            n_heads=4
        )


        self.pooling_type = pooling_type
        self.active_site_embedding = nn.Embedding(num_embeddings=2, embedding_dim=message_dim)
        #self.site_embeddings = nn.Embedding(num_embeddings=21, embedding_dim=message_dim)
        self.element_embeddings = nn.Embedding(num_embeddings=len(element_pool), embedding_dim=message_dim)
        self.hidden_layers = nn.ModuleList(hidden_layers) #+ [output_layer]
        self.hidden_dim_rep = hidden_dim_rep
        self.n_hidden_layers = n_hidden_layers
        self.message_dim = message_dim
        self.aggr = aggr
    
    def embedding_block(self, batch, time, drop_condition):
        x_t, conds, batch_indices, active_sites, active_site_dists = (
            batch.x, 
            batch.y, 
            batch.batch, 
            batch.active_sites,
            batch.active_site_dists
        )
        #Embed the atomic species via learnable embeddings
        indices = torch.argmax(x_t, dim=1)
        x_t = self.element_embeddings(indices)

        #Embed geometrical information via active site embeddings
        #indices = torch.arange(0, 21, 1, device=self.device).repeat(batch.batch_size)
        active_site_emb = self.active_site_embedding(active_sites)
        active_site_dist_emb = self.dist_embedder(active_site_dists)
        geom_rep = torch.hstack([active_site_emb, active_site_dist_emb])

        #Embed time via sinusoidal embedding, and expand to each node in every graph
        time_embedded = self.get_time_embedding(time=time)[batch_indices]

        #Embed the condition via the given conditioning module, and expand to each node in every graph
        if self.cond_embedder is None:
            embedded_conds = None
        else:
            embedded_conds = self.cond_embedder.forward(condition=conds, drop_condition=drop_condition)[batch_indices]

        return x_t, geom_rep, embedded_conds, time_embedded

    def representation(self, x_t, batch, time, drop_condition:bool):
        x_t, geom_rep, embedded_conds, time_embedded = self.embedding_block(batch=batch, time=time, drop_condition=drop_condition)
        edge_index = batch.edge_index
        #print(x_t.shape, geom_rep.shape, embedded_conds.shape, time_embedded.shape)
        x_t += self.input_layer.forward(
            x_t=x_t, 
            geom_rep=geom_rep,
            edge_index=edge_index, 
            conds_embedded=embedded_conds, 
            time_embedded=time_embedded,
        )
        for message_passing_layer in self.hidden_layers:
            x_t += message_passing_layer.forward(
                x_t=x_t,
                geom_rep=geom_rep, 
                edge_index=edge_index, 
                conds_embedded=embedded_conds, 
                time_embedded=time_embedded
            )
        return x_t, embedded_conds
    
    def get_logits(self, x_t, batch, time, drop_condition):
        x_t, embedded_conds = self.representation(
            x_t=x_t,
            batch=batch,
            time=time,
            drop_condition=drop_condition
        )
        ctx = self.dist_pooling(x_t, pos=batch.pos, batch=batch.batch)
        if self.pooling is not None:
            x_t_global = self.pooling(x=x_t, batch=batch.batch)
            if embedded_conds is not None:
                return self.prob_head(torch.hstack([x_t, ctx, x_t_global[batch.batch], embedded_conds]))
            else:
                return self.prob_head(torch.hstack([x_t, ctx, x_t_global[batch.batch]]))
        else:
            if embedded_conds is not None:
                return self.prob_head(torch.hstack([x_t, ctx, embedded_conds]))
            else:
                return self.prob_head(x_t)#self.prob_head(torch.hstack([x_t, ctx]))

    def get_rate_pred(self, x0, batch, time):
        x_t, _ = self.representation(
            x_t=x0,
            batch=batch,
            time=time,
            drop_condition=True
        )
        x_t_pooled = global_add_pool(x_t, batch.batch)
        reaction_rate = self.rate_head(x_t_pooled)
        return reaction_rate

    def get_sample_loader(self, dataset, batch_size, shuffle:bool=True):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def get_sample_dataset_from_atoms_list(self, atoms_list:list, condition_key:str, dataset_kwargs:dict={}):
        dataset = get_dataset_from_atoms_list(
            atoms_list=atoms_list,
            element_pool=self.element_pool,
            condition_key=condition_key,
            graph_kwargs=dataset_kwargs
        )
        return dataset

    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        denoiser_info = {
            "denoiser_type":"DiscreteGNNDenoiser",
            "message_dim":self.message_dim,
            "n_hidden_layers":self.n_hidden_layers,
            "hidden_dim_rep":self.hidden_dim_rep,
            "time_embedding_dim":self.time_embedding_dim,
            "aggr":self.aggr,
            "pooling_type":self.pooling_type
        }
        state_dict.update(denoiser_info)
        return state_dict
