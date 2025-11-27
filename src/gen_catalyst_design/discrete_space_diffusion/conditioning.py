import torch.nn.functional as F
import torch.nn as nn
import torch

# -------------------------------------------------------------------------------------
# EMBEDDING CONDITIONS BASE-CLASS
# -------------------------------------------------------------------------------------


class ConditioningEmbedder(nn.Module):
    def __init__(self, cond_dim:int=1, embedding_dim:int=28):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        self.uncond_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)

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


