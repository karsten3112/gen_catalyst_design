import torch.nn.functional as F
import torch.nn as nn
import torch

class Conditioning(nn.Module):
    def __init__(self, embedding_dim:int=32, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.none_embedding = nn.Embedding(
            num_embeddings=1, 
            embedding_dim=embedding_dim
        )

    def set_device(self, device):
        self.device = device
    
    def get_state_dict(self):
        return {"embedding_dim":self.embedding_dim}

    def sinusoidal_embedding(self, pos_encoding):
        indices = torch.arange(0, self.embedding_dim, 1, device=self.device)
        angle_rates = 1 / torch.pow(10000, (2 * (indices // 2)) / self.embedding_dim)
        freq_encoding = pos_encoding[:,None]*angle_rates[None,:]
        sinusoidal = torch.zeros_like(freq_encoding, device=self.device)
        sinusoidal[:, 0::2] = torch.sin(freq_encoding[:, 0::2])
        sinusoidal[:, 1::2] = torch.cos(freq_encoding[:, 1::2])
        return sinusoidal
    
    def get_condition_embedding(self, condition, batch_size, drop_condition:bool):
        if drop_condition is True:
            return self.none_embedding(torch.tensor([0], device=self.device)).repeat(batch_size, 1)
        else:
            return self.embed_condition(condition=condition)
    
    def embed_condition(self, condition):
        raise NotImplementedError("Has to be implemented by sub-class")


class NoneConditioning(Conditioning):
    def __init__(self):
        super().__init__(embedding_dim=0, device=None)

    def get_state_dict(self):
        state_dict = {"conditioning_type":"None"}
        return state_dict

    def get_condition_embedding(self, condition, batch_size, drop_condition):
        return None


class RateClassification(Conditioning):
    def __init__(self, embedding_dim = 32, device=None):
        super().__init__(embedding_dim, device)




class RateConditioning(Conditioning):
    def __init__(self, embedding_dim = 32, activation_func:callable=nn.ReLU(), device=None):
        super().__init__(embedding_dim, device)
        self.ml_layers = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            activation_func,
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        )
    
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"conditioning_type":"RateConditioning"})
        return state_dict

    def embed_condition(self, condition):
        rate_embedded = self.sinusoidal_embedding(pos_encoding=condition)
        return self.ml_layers(rate_embedded)