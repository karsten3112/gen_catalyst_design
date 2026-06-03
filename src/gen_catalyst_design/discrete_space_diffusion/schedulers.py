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
            time_sample_method:str="stratified"
        ):
        super().__init__()
        if t_init < 1:
            raise Exception(f"initial time stamp t_init, cannot be less than 1; given was:{t_init}")
        self.t_init = t_init
        self.t_final = t_final
        self.device = None
        self.time_sample_method = time_sample_method

    def get_state_dict(self):
        state_dict = {
            "t_init":self.t_init,
            "t_final":self.t_final,
            "time_sample_method":self.time_sample_method
        }
        return state_dict

    def set_device(self, device):
        self.device = device

    def __call__(self, t:torch.tensor):
        raise Exception("Must be implemented by sub-class")
    
    def sample_time(self, n_samples, t_span:tuple=None):
        methods = {
            "uniform": self.sample_time_uniformly, 
            "stratified": self.sample_time_stratified
        }
        if self.time_sample_method not in methods:
            raise Exception(f"method for sampling time: {self.time_sample_method} is not implemented")
        else:
            return methods[self.time_sample_method](n_samples=n_samples, t_span=t_span)
    
    def sample_time_uniformly(self, n_samples:int, t_span:tuple=None):
        if t_span is not None:
            low, high = t_span[0], t_span[1]
        else:
            low, high = self.t_init, self.t_final
        return torch.randint(low=low, high=high, size=(n_samples,), device=self.device) 
    
    def sample_time_stratified(self, n_samples:int, t_span:tuple=None):
        if t_span is not None:
            low, high = t_span[0], t_span[1]
        else:
            low, high = self.t_init, self.t_final
        edges = torch.linspace(low, high + 1, n_samples + 1, device=self.device)  # +1 to make upper edge exclusive
        lows = edges[:-1].long()
        highs = edges[1:].long()
        u = torch.rand(n_samples, device=self.device)
        t = (lows + (u * (highs - lows).float()).floor().long())
        t = t.clamp(min=low, max=high)
        t = t[torch.randperm(n_samples, device=self.device)]
        return t


# -------------------------------------------------------------------------------------
# SCHEDULES DERIVED FROM EVOLUTION OF BETA
# -------------------------------------------------------------------------------------

class DiscreteBetaScheduler(DiscreteTimeScheduler):
    def __init__(
            self,
            beta_min:float=1e-3,
            beta_max:float=1.0, 
            t_init = 1, 
            t_final = 1000, 
            time_sample_method = "stratified"
        ):
        super().__init__(t_init, t_final, time_sample_method)
        self.beta_min = torch.tensor(beta_min, dtype=torch.float64)
        self.beta_max = torch.tensor(beta_max, dtype=torch.float64)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"beta_min":self.beta_min.item(), "beta_max":self.beta_max.item()})
        return state_dict


class LinearBetaScheduler(DiscreteBetaScheduler):
    def __init__(self, beta_min = 0.001, beta_max = 1, t_init=1, t_final=1000, time_sample_method="stratified", **kwargs):
        super().__init__(beta_min, beta_max, t_init, t_final, time_sample_method)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"scheduler_type":"LinearBetaScheduler"})
        return state_dict

    def __call__(self, t:torch.tensor):
        return self.beta_min + (t - self.t_init)/(self.t_final-self.t_init)*(self.beta_max - self.beta_min)


class ExponentialBetaScheduler(DiscreteBetaScheduler):
    def __init__(self, beta_min = 0.001, beta_max = 1, t_init=1, t_final=1000, time_sample_method="stratified", **kwargs):
        super().__init__(beta_min, beta_max, t_init, t_final, time_sample_method)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"scheduler_type":"ExponentialBetaScheduler"})
        return state_dict

    def __call__(self, t):
        return self.beta_min*torch.pow(self.beta_max/self.beta_min, (t-1.0)/(self.t_final-1.0))

# -------------------------------------------------------------------------------------
# SCHEDULES DERIVED FROM EVOLUTION OF ALPHA
# -------------------------------------------------------------------------------------

class DiscreteAlphaScheduler(DiscreteTimeScheduler):
    def __init__(self, t_init = 1, t_final = 1000, time_sample_method = "stratified"):
        super().__init__(t_init, t_final, time_sample_method)


class CosineScheduler(DiscreteAlphaScheduler):
    def __init__(self, reg:float=1e-1, t_init=1, t_final=1000, time_sample_method="stratified", **kwargs):
        super().__init__(t_init, t_final, time_sample_method)
        self.reg = reg
    
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"scheduler_type":"CosineScheduler", "reg":self.reg})
        return state_dict
   
    def set_device(self, device):
        #self.reg = self.reg.to(device=device)
        return super().set_device(device)

    def cos(self, t:torch.tensor):
        return torch.cos((t/self.t_final+self.reg)/(1.0+self.reg)*torch.pi/2.0)**2

    def alpha_t(self, t):
        return self.cos(t)/self.cos(torch.tensor(0, device=self.device))

    def __call__(self, t):
        return 1.0 - self.alpha_t(t=t)/self.alpha_t(t=(t-1))
    
class LinearAlphaScheduler(DiscreteAlphaScheduler):
    def __init__(self, t_init=1, t_final=1000, time_sample_method="stratified", **kwargs):
        super().__init__(t_init, t_final, time_sample_method)
        self.alpha_t_final = torch.tensor(0.0)
        self.alpha_t_init = torch.tensor(1.0)

    def alpha_t(self, t):
        return self.alpha_t_init + t/self.t_final*(self.alpha_t_final-self.alpha_t_init)
    
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update({"scheduler_type":"LinearAlphaScheduler"})
        return state_dict

    def __call__(self, t):
        return 1.0 - self.alpha_t(t=t)/self.alpha_t(t=(t-1))