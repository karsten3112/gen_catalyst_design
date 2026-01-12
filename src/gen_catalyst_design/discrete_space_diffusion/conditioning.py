import torch.nn.functional as F
import torch.nn as nn
import torch

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
    def __init__(self, num_labels:int, embedding_dim = 28):
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
    def __init__(self, cond_dim = 1, embedding_dim = 28, activation_function=torch.nn.ReLU()):
        super().__init__(cond_dim, embedding_dim)
        self.ml_layers = nn.Sequential(
            nn.Linear(in_features=self.cond_dim, out_features=self.embedding_dim),
            activation_function,
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        )
    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"embedding_type":"RateEmbedder"})
        return state_dict

    def get_embedded_condition(self, condition):
        return self.ml_layers(condition.view(-1,1))
    
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
        print(rate_class)
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

