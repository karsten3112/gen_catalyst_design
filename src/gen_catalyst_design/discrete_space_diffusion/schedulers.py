import torch
import torch.nn as nn

# -------------------------------------------------------------------------------------
# DISCRETE TIME SCHEDULER BASE CLASS
# -------------------------------------------------------------------------------------


class DiscreteTimeScheduler(nn.Module):
    def __init__(
            self, 
            t_init:int=1, 
            t_final:int=1000, 
            beta_max:float=1.0,
            beta_min:float=1e-3,
        ):
        super().__init__()
        if t_init < 1:
            raise Exception(f"initial time stamp t_init, cannot be less than 1; given was:{t_init}")
        self.t_init = t_init
        self.t_final = t_final
        self.beta_max = torch.tensor(beta_max)
        self.beta_min = torch.tensor(beta_min)
        self.device = None

    @property
    def const_state_dict(self):
        state_dict = {
            "t_init":self.t_init,
            "t_final":self.t_final,
            "beta_max":self.beta_max.item(),
            "beta_min":self.beta_min.item()
        }
        return state_dict

    def set_device(self, device):
        self.device = device

    def __call__(self, t:torch.tensor):
        raise Exception("Must be implemented by sub-class")
    
    def sample_time(self, n_samples:int):
        return torch.randint(low=self.t_init, high=self.t_final, size=(n_samples,), device=self.device)


# -------------------------------------------------------------------------------------
# LINEAR SCHEDULER
# -------------------------------------------------------------------------------------


class LinearScheduler(DiscreteTimeScheduler):
    def __init__(self, t_init = 1, t_final = 1000, beta_max = 1, beta_min = 0.001):
        super().__init__(t_init, t_final, beta_max, beta_min)

    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"scheduler_type":"LinearScheduler"})
        return state_dict

    def __call__(self, t:torch.tensor):
        return self.beta_min + (t - self.t_init)/(self.t_final-self.t_init)*(self.beta_max - self.beta_min)


# -------------------------------------------------------------------------------------
# COSINE SCHEDULER
# -------------------------------------------------------------------------------------


class CosineScheduler(DiscreteTimeScheduler):
    def __init__(self, t_init = 1, t_final = 1000, beta_max = 1, beta_min = 0.001, reg:float=1e-3):
        super().__init__(t_init, t_final, beta_max, beta_min)
        self.reg = torch.tensor(reg)
    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"scheduler_type":"CosineScheduler"})
        return state_dict

    def cos(self, t:torch.tensor):
        return torch.cos((t/self.t_final+self.reg)/(1.0+self.reg)*torch.pi/2.0)**2

    def alpha_t(self, t):
        return self.cos(t)/self.cos(torch.tensor(0))

    def __call__(self, t):
        return 1.0 - self.alpha_t(t=t)/self.alpha_t(t=(t-1))


# -------------------------------------------------------------------------------------
# EXPONENTIAL SCHEDULER
# -------------------------------------------------------------------------------------  


class ExponentialScheduler(DiscreteTimeScheduler):
    def __init__(self, t_init = 1, t_final = 1000, beta_max = 1, beta_min = 0.001):
        super().__init__(t_init, t_final, beta_max, beta_min)

    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"scheduler_type":"ExponentialScheduler"})
        return state_dict

    def __call__(self, t):
        return self.beta_min*torch.pow(self.beta_max/self.beta_min, (t-1.0)/(self.t_final-1.0))
    