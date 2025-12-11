import torch
from torch.distributions import Categorical
from .schedulers import DiscreteTimeScheduler
import torch.nn.functional as F
import torch.nn as nn


# -------------------------------------------------------------------------------------
# DISCRETE SPACE NOISER BASE CLASS
# -------------------------------------------------------------------------------------

class DiscreteSpaceNoiser(nn.Module):
    def __init__(
            self, 
            element_pool:list, 
            accumulated_q_matrices:torch.tensor=None, 
            random_state:int=42
        ):
        super().__init__()
        self.accumulated_q_matrices = accumulated_q_matrices
        self.random_state = random_state
        self.n_classes = len(element_pool)
        self.stationary_dist = None
        self.device = None

    @property
    def const_state_dict(self):
        return {}

    def set_device(self, device):
        self.device = device
        self.accumulated_q_matrices = self.accumulated_q_matrices.to(device=device)

    def pre_compute_accum_q_matrices(self, scheduler:DiscreteTimeScheduler):
        if self.accumulated_q_matrices is None:
            time = torch.arange(scheduler.t_init, scheduler.t_final+1, 1)
            beta_t_batch = scheduler(t=time)
            Qts = self.__call__(beta_t_batch=beta_t_batch)
            result_matrix = torch.eye(n=self.n_classes)
            regularized_matrix = result_matrix*1.0
            accum_matrices = [regularized_matrix]
            for Qt in Qts: 
                result_matrix @= Qt
                accum_matrices.append(result_matrix)
            self.accumulated_q_matrices = torch.stack(accum_matrices)

    def __call__(self, beta_t_batch:torch.tensor) -> torch.tensor:
        raise Exception ("Must be implemented in sub-class")

    def get_transition_probabilities(self, x_t_batch:torch.tensor, time_batch:torch.tensor, scheduler:DiscreteTimeScheduler):
        beta_t_batch = scheduler(t=time_batch)
        Qts = self.__call__(beta_t_batch=beta_t_batch)
        probs = torch.bmm(x_t_batch.unsqueeze(1), Qts).squeeze()
        return probs
    
    def get_accum_transition_probabilities(self, x0_batch:torch.tensor, time_batch:torch.tensor):
        Q_accum_t = self.accumulated_q_matrices[time_batch]
        probs = torch.bmm(x0_batch.unsqueeze(1), Q_accum_t).squeeze()
        return probs

    def get_reverse_transition_probabilities(
            self, 
            x_t_batch:torch.tensor, 
            time_batch:torch.tensor, 
            x0_batch:torch.tensor,
            scheduler:DiscreteTimeScheduler
        ):
        if self.accumulated_q_matrices is None:
            raise Exception("reverse probabilites cannot be calculated before the accumulated matrices have been calculated")
        else:
            Q_accum_t = self.accumulated_q_matrices[time_batch]
            Q_accum_tm1 = self.accumulated_q_matrices[time_batch-1]
            beta_t_batch = scheduler(t=time_batch)
            Qts = self.__call__(beta_t_batch=beta_t_batch)
            p1 = torch.bmm(x_t_batch.unsqueeze(1), Qts.transpose(-2,-1)).squeeze()
            p2 = torch.bmm(x0_batch.unsqueeze(1), Q_accum_tm1).squeeze() #(1.0*x0) @ Q_accum_tm1
            den = torch.bmm(x0_batch.unsqueeze(1), Q_accum_t).squeeze() #(1.0*x0) @ Q_accum_t
            denom = (den * x_t_batch).sum(dim=-1, keepdim=True)
            reg_indices = (denom > 0.0).reshape(shape=(len(denom),))
            probs = p1*p2
            probs[reg_indices]/=denom[reg_indices]
            return probs
    
    def get_dist(self, probabilites:torch.tensor) -> Categorical:
        return Categorical(probs=probabilites)

    def noise_batch_x0_xt(self, batch, time_batch:torch.tensor):
        probs = self.get_accum_transition_probabilities(x0_batch=batch.x*1.0, time_batch=time_batch)
        noised_xs = self.sample_transition(probabilites=probs)
        #print(batch.x.device)
        batch.x = noised_xs
        #print(batch.x.device)
        try:
            x_stacked = torch.hstack([batch.x, batch.active_sites])
            batch.edge_attr = x_stacked[batch.edge_index[0]] + x_stacked[batch.edge_index[1]]
        except:
            pass

    def noise_x0_xt(self, x0_batch:torch.tensor, time_batch:torch.tensor, ):
        probs = self.get_accum_transition_probabilities(x0_batch=x0_batch, time_batch=time_batch)
        noised_x_batch = self.sample_transition(probabilites=probs)
        return noised_x_batch

    def noise_step(self, x_t:torch.tensor, time:torch.tensor):
        probs = self.get_transition_probabilities(x_t=x_t, time=time)
        noised_x = self.sample_transition(probabilites=probs)
        return noised_x

    def sample_transition(self, probabilites:torch.tensor, **kwargs):
        dist = self.get_dist(probabilites)
        sample = F.one_hot(dist.sample(), num_classes=self.n_classes)
        return sample
    
    def sample_reverse_transition(self, probabilites:torch.tensor, **kwargs):
        raise Exception("Must be implemented by sub-class")

    def sample_from_stationary(self, num_atoms:int):
        probs = self.stationary_dist*torch.ones(size=(num_atoms,1))
        sample = self.sample_transition(probabilites=probs)
        return sample


# -------------------------------------------------------------------------------------
# ABSORBING-STATE NOISER
# -------------------------------------------------------------------------------------


class UniformTransitionsNoiser(DiscreteSpaceNoiser):
    def __init__(self, element_pool, accumulated_q_matrices = None, random_state = 42):
        super().__init__(element_pool, accumulated_q_matrices, random_state)
        self.stationary_dist = self.get_stationary_dist()
    
    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"noiser_type":"UniformTransitionsNoiser"})
        return state_dict

    def __call__(self, beta_t_batch:torch.tensor):
        n_classes = self.n_classes
        beta_t_reshaped = beta_t_batch[:, None, None]
        t1 = (1.0-beta_t_reshaped)*torch.eye(n=n_classes,dtype=torch.long, device=self.device)
        t2 = beta_t_reshaped/n_classes*torch.ones(size=(n_classes,n_classes),dtype=torch.long, device=self.device)
        return t1 + t2
    
    def get_stationary_dist(self):
        return torch.ones(size=(self.n_classes,))/self.n_classes

    def sample_reverse_transition(self, probabilites, **kwargs):
        return super().sample_transition(probabilites=probabilites)


# -------------------------------------------------------------------------------------
# UNIFORM-TRANSITIONS NOISER
# -------------------------------------------------------------------------------------


class AbsorbingStateNoiser(DiscreteSpaceNoiser):
    def __init__(self, element_pool, accumulated_q_matrices = None, random_state = 42, eps=1e-12):
        if "(X)" not in element_pool:
            raise Exception(f"Absorbing state (X) not found in element pool being: {element_pool}")
        super().__init__(element_pool, accumulated_q_matrices, random_state)
        for i, elem in enumerate(element_pool):
            if elem == "(X)":
                self.absorbing_state_index = i
                break
        self.stationary_dist = self.get_stationary_dist()
        self.eps = eps

    @property
    def const_state_dict(self):
        state_dict = super().const_state_dict
        state_dict.update({"noiser_type":"AbsorbingStateNoiser"})
        return state_dict

    def get_stationary_dist(self):
        probs = torch.zeros(self.n_classes)
        probs[self.absorbing_state_index]+=1.0
        return probs

    def __call__(self, beta_t_batch):
        n_classes = self.n_classes
        beta_t_reshaped = beta_t_batch[:, None, None]
        t1 = (1.0-beta_t_reshaped)*torch.eye(n=n_classes,dtype=torch.long, device=self.device)
        e_m = (F.one_hot(torch.tensor(self.absorbing_state_index, device=self.device), num_classes=n_classes)*1.0).unsqueeze(0)
        t2 = beta_t_reshaped*(torch.ones(size=(n_classes,1), device=self.device) @ e_m)
        return t1 + t2
    
    def sample_reverse_transition(self, probabilites, time:torch.tensor):
        if time == 1:
            probabilites[:,self.absorbing_state_index] = 0.0 #enforce 0 probability for absorbing state at initial time-step
        return super().sample_transition(probabilites)
