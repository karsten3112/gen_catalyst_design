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
    
    @property
    def const_state_dict(self):
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

    @property
    def const_state_dict(self):
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
    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"conditioning_type":"RateConditioning"})
        return state_dict

    def embed_condition(self, condition):
        rate_embedded = self.sinusoidal_embedding(pos_encoding=condition)
        return self.ml_layers(rate_embedded)
    


# -------------------------------------------------------------------------------------
# EMBEDDING CONDITIONS BASE-CLASS
# -------------------------------------------------------------------------------------

class ConditioningEmbedder(nn.Module):
    def __init__(self, cond_dim:int=1, embedding_dim:int=28, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        self.uncond_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.device = device

    def set_device(self, device):
        self.device = device
    
    @property
    def const_state_dict(self):
        return {"cond_dim":self.cond_dim, "embedding_dim":self.embedding_dim}

    def get_embedded_condition(self, condition:torch.tensor):
        raise Exception("Must be implemented by sub-class")

    def forward(self, condition:torch.tensor, drop_condition:bool=False):
        if drop_condition:
            return self.uncond_embedding(torch.zeros_like(condition, dtype=torch.long))
        else:
            return self.get_embedded_condition(condition=condition)

    def get_condition_from_data_dict(self, data_dict:dict):
        raise NotImplementedError("Must be implemented by-subclass")

# -------------------------------------------------------------------------------------
# CLASS LABEL EMBEDDING CLASS
# -------------------------------------------------------------------------------------

class ClassLabelEmbedder(ConditioningEmbedder):
    def __init__(self, num_labels:int=2, embedding_dim = 28):
        super().__init__(embedding_dim=embedding_dim)
        self.learned_embs = nn.Embedding(num_embeddings=num_labels, embedding_dim=self.embedding_dim)
        self.num_labels = num_labels

    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.pop("cond_dim")
        state_dict["num_labels"] = self.num_labels
        state_dict.update({"embedding_type":"ClassLabelEmbedder"})
        return state_dict
    
    def get_embedded_condition(self, condition):
        return self.learned_embs(condition)
    
    def get_condition_from_data_dict(self, data_dict):
        class_type = data_dict["class"]
        return torch.tensor(class_type)


# -------------------------------------------------------------------------------------
# RATE EMBEDDING CLASS
# -------------------------------------------------------------------------------------


class RateEmbedder(ConditioningEmbedder):
    def __init__(self, cond_dim = 1, embedding_dim = 28, activation_function=torch.nn.ReLU(), device=None):
        super().__init__(cond_dim, embedding_dim, device=device)
        self.ml_layers = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            activation_function,
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        )
    
    def get_rate_embedding(self, rate):
        indices = torch.arange(0, self.embedding_dim, 1, device=self.device)
        angle_rates = 1 / torch.pow(10000, (2 * (indices // 2)) / self.embedding_dim)
        rates=rate[:,None]*angle_rates[None,:]
        rate_embedded = torch.zeros_like(rates, device=self.device)
        rate_embedded[:, 0::2] = torch.sin(rates[:, 0::2])
        rate_embedded[:, 1::2] = torch.cos(rates[:, 1::2])
        return rate_embedded

    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"embedding_type":"RateEmbedder"})
        return state_dict

    def get_embedded_condition(self, condition):
        rate_embedding = self.get_rate_embedding(rate=condition)
        return self.ml_layers(rate_embedding)
    
    def get_condition_from_data_dict(self, data_dict):
        return torch.tensor(data=data_dict["rate"])
        

# -------------------------------------------------------------------------------------
# RATE CLASSIFICATION EMBEDDING CLASS
# -------------------------------------------------------------------------------------

class RateClassEmbedder(ClassLabelEmbedder):
    def __init__(self, num_labels, embedding_dim=28):
        super().__init__(num_labels, embedding_dim)
        self.rate_embedder = RateEmbedder(cond_dim=1, embedding_dim=num_labels)

    def get_embedded_condition(self, condition):
        embedded_rate = F.softmax(self.rate_embedder(condition))
        rate_class = torch.argmax(embedded_rate)
        return super().get_embedded_condition(rate_class)
    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.pop("cond_dim")
        state_dict["num_labels"] = self.num_labels
        state_dict.update({"embedding_type":"RateClassEmbedder"})
        return state_dict


class ActiveSiteConditioning(ClassLabelEmbedder):
    def __init__(self, element_pool:list, num_active_sites:int=4, embedding_dim = 28, activation_function=torch.nn.ReLU()):
        super().__init__(num_labels=len(element_pool), embedding_dim=embedding_dim)
        self.element_pool = element_pool
        self.elem_to_int = {element:i for i, element in enumerate(element_pool)}
        self.ml_layers = nn.Sequential(
            nn.Linear(in_features=num_active_sites*self.embedding_dim, out_features=self.embedding_dim),
            activation_function,
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        )

    def get_embedded_condition(self, condition):
        embedded_conditions = []
        for cond in condition:
            idx = torch.tensor([self.elem_to_int[elem] for elem in cond], dtype=torch.long, device=self.device)
            learned_embs = super().get_embedded_condition(idx)
            embedded_condition = self.ml_layers(torch.hstack(list(learned_embs)))
            embedded_conditions.append(embedded_condition)
        return torch.vstack(embedded_conditions)
    
    def forward(self, condition, drop_condition = False):
        if drop_condition:
            return self.uncond_embedding(torch.zeros(size=(len(condition), ), dtype=torch.long, device=self.device))
        else:
            return self.get_embedded_condition(condition=condition)

    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.pop("num_labels")
        state_dict.update({"embedding_type":"ActiveSiteConditioning"})
        state_dict["element_pool"] = self.element_pool
        return state_dict

