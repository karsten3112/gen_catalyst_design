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


class NoneConditioning(nn.Module):
    def __init__(self, embedding_dim=64, device=None, **kwargs):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.none_embedding = torch.zeros(size=(self.embedding_dim, ), device=self.device)

    def set_device(self, device):
        self.device = device
        self.none_embedding = self.none_embedding.to(device=device)

    def get_state_dict(self):
        state_dict = {
            "conditioning_type":"None",
            "embedding_dim":self.embedding_dim
        }
        return state_dict

    def get_condition_embedding(self, condition, batch_size, drop_condition):
        return self.none_embedding.repeat(batch_size,1)


class RateScalarConditioning(Conditioning):
    def __init__(self, embedding_dim = 32, activation_func:callable=nn.ReLU(), device=None):
        super().__init__(embedding_dim, device)
        #self.linear_embed = nn.Linear(in_features=1, out_features=self.embedding_dim)
        self.ml_layers = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            activation_func,
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        )
    
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"conditioning_type":"RateScalarConditioning"})
        return state_dict

    def embed_condition(self, condition):
        rate_embedded = self.sinusoidal_embedding(pos_encoding=condition)
        return self.ml_layers(rate_embedded)

class RateMantissaConditioning(Conditioning):
    def __init__(self, embedding_dim = 32, apply_log:bool=False, activation_func:callable=nn.ReLU(),device=None):
        super().__init__(embedding_dim, device)
        self.apply_log = apply_log
        self.ml_layers = nn.Sequential(
            nn.Linear(in_features=2*self.embedding_dim, out_features=self.embedding_dim),
            activation_func,
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        )

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"conditioning_type":"RateMantissaConditioning"})
        return state_dict

    def embed_condition(self, condition):
        if self.apply_log:
            exponents = torch.floor(torch.log10(condition))
            mags = condition/(10**exponents)
        else:
            exponents = torch.floor(condition)
            mags = (10**condition)/(10**exponents)
        
        mag_embedded = self.sinusoidal_embedding(pos_encoding=mags)
        exp_embedded = self.sinusoidal_embedding(pos_encoding=exponents)
        return self.ml_layers(torch.hstack([mag_embedded, exp_embedded]))



class EformConditioning(RateScalarConditioning):
    def __init__(self, embedding_dim=32, activation_func = nn.ReLU(), device=None):
        super().__init__(embedding_dim, activation_func, device)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict["conditioning_type"] = "EformConditioning"
        return state_dict
    


class RateClassConditioning(Conditioning):
    def __init__(
            self,
            rate_min:float=0.0,
            rate_max:float=1e3,
            num_classes:int=20, 
            embedding_dim = 64, 
            device=None
        ):
        super().__init__(embedding_dim, device)
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.num_classes = num_classes
        self.class_embeddings = nn.Embedding(
            num_embeddings=num_classes, 
            embedding_dim=embedding_dim
        )
        self.class_divisions = torch.linspace(rate_min, rate_max, num_classes)
        self.spacing = torch.diff(self.class_divisions)[0]
        self.mixing_mlp = nn.Sequential(
            nn.Linear(2*self.embedding_dim, 2*self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2*self.embedding_dim, self.embedding_dim),
        )
    
    def set_device(self, device):
        super().set_device(device)
        self.spacing = self.spacing.to(device=device)
        self.class_divisions = self.class_divisions.to(device=device)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({
                "conditioning_type":"RateClassConditioning",
                "rate_min":self.rate_min,
                "rate_max":self.rate_max,
                "num_classes":self.num_classes
        })
        return state_dict
    
    def embed_condition(self, condition):
        class_indices = torch.bucketize(input=condition, boundaries=self.class_divisions)
        lin_scaled_rates = (condition-self.class_divisions[class_indices-1])/self.spacing
        embedded_classes = self.class_embeddings(class_indices)
        embedded_rate = self.sinusoidal_embedding(pos_encoding=lin_scaled_rates)
        features = torch.hstack([embedded_classes, embedded_rate])
        return self.mixing_mlp(features)