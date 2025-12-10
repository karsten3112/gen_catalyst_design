from .schedulers import DiscreteTimeScheduler, ExponentialScheduler, CosineScheduler, LinearScheduler
from .noisers import DiscreteSpaceNoiser, UniformTransitionsNoiser, AbsorbingStateNoiser
from .denoisers import DiscreteSpaceDenoiser, DiscreteGNNDenoiser
from .conditioning import RateEmbedder, ClassLabelEmbedder, RateClassEmbedder
from ase.atoms import Atoms
from .Dataset import get_elements_from_onehots
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ase.atoms import Atoms
from torch.distributions import Categorical


class DiffusionModel(LightningModule):
    def __init__(
            self,
            element_pool:list=[],
            scheduler:DiscreteTimeScheduler=None,
            noiser:DiscreteSpaceNoiser=None,
            denoiser:DiscreteSpaceDenoiser=None,
            drop_prob:float=0.2,
            random_seed:int=42,
            lr:float=1e-4,
            weight_decay:float=1e-4,
            use_x0_reparam:bool=False,
            auxillary_weight:float=None
        ):
        super().__init__()
        self.element_pool = element_pool
        self.scheduler = scheduler
        self.noiser = noiser
        self.denoiser = denoiser
        self.random_seed = random_seed
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay
        if self.noiser is not None:
            if self.noiser.accumulated_q_matrices is None:
                self.noiser.pre_compute_accum_q_matrices(scheduler=self.scheduler)

        self.cross_entropy_logits = nn.CrossEntropyLoss()
        #set to false for now as it does not work currently
        self.use_x0_reparam = use_x0_reparam
        self.auxillary_weight = auxillary_weight

    def on_fit_start(self):
        device = self.device
        self.noiser.set_device(device=device)
        self.scheduler.set_device(device=device)
    
    #Does not work currently
    def perform_x0_reparam(self, denoise_logits, x_t, batch, time):
        denoise_probs = torch.softmax(denoise_logits, dim=-1)
        x0s = [F.one_hot(torch.tensor(i), num_classes=len(self.element_pool))*torch.ones(size=(len(x_t), 1)) for i in range(len(self.element_pool))]
        print(x_t[0])
        q_revs_tot = torch.stack([self.noiser.get_reverse_transition_probabilities(
            x0_batch=x0*1.0,
            x_t_batch=x_t*1.0, 
            time_batch=time[batch.batch], 
            scheduler=self.scheduler
        ) for x0 in x0s
        ])
        reverse_probs = (denoise_probs[None, :, :]*q_revs_tot).sum(dim=0)
        normalized_probs = reverse_probs/reverse_probs.sum(dim=1, keepdim=True)
        return normalized_probs
    
    def get_reverse_transition_probabilities(self, x_t, batch, time, guidance_scale):
        logits = [
           self.denoiser.get_logits(
               x_t=x_t,
               batch=batch,
               time=time,
               scheduler=self.scheduler,
               drop_cond=drop_condition
           )
           for drop_condition in [False, True]
        ]
        
        #Use classifier free guidance in log-space
        guided_logits = guidance_scale*logits[0] + (1.0-guidance_scale)*logits[1]
        if self.use_x0_reparam:
            guided_probs = self.perform_x0_reparam(
                denoise_logits=guided_logits,
                x_t=x_t*1.0,
                batch=batch,
                time=time
            )
            return guided_probs
        else:
            return F.softmax(guided_logits, dim=-1)


    def get_auxillary_term_loss(self, batch, drop_condition):
        #Calculate the auxillary term as mentioned in D3PM paper
        t_span = (1, self.scheduler.t_final)
        time = self.scheduler.sample_time(n_samples=batch.batch_size, t_span=t_span)
        
        x_t = self.noiser.noise_x0_xt(x0_batch=batch.x*1.0, time_batch=time[batch.batch])
        
        logits = self.denoiser.get_logits(
            x_t=x_t*1.0,
            batch=batch,
            time=time,
            scheduler=self.scheduler,
            drop_cond=drop_condition   
        )
        q_forward = self.noiser.get_transition_probabilities(
            x_t_batch=x_t*1.0,
            time_batch=time[batch.batch],
            scheduler=self.scheduler
        )
        return self.cross_entropy_logits(logits, q_forward)

    def calculate_cross_entropy_from_probs(self, p_dist, q_dist):
        mask_indices = q_dist > 0.0
        return -(p_dist[mask_indices]*torch.log(q_dist[mask_indices])).mean()


    def get_denoise_matching_term_loss(self, batch, drop_condition):
        #Calculate the known true posterier for when x0 is known
        t_span = (2, self.scheduler.t_final)
        time = self.scheduler.sample_time(n_samples=batch.batch_size, t_span=t_span)
        
        x_t = self.noiser.noise_x0_xt(x0_batch=batch.x*1.0, time_batch=time[batch.batch])

        q_revs = self.noiser.get_reverse_transition_probabilities(
            x0_batch=batch.x*1.0,
            x_t_batch=x_t*1.0, 
            time_batch=time[batch.batch], 
            scheduler=self.scheduler
        )

        logits = self.denoiser.get_logits(
            x_t=x_t*1.0,
            batch=batch,
            time=time,
            scheduler=self.scheduler,
            drop_cond=drop_condition   
        )

        #Apply the x0 reparameterization as outlined in D3PM if desired
        #return the cross entropy between the true posterier and the predicted reversals
        #Note here that this is equal to the KL-divergence up to a constant which is not learnable
        if self.use_x0_reparam:
            denoise_probs = self.perform_x0_reparam(
                denoise_logits=logits,
                x_t=x_t*1.0,
                batch=batch,
                time=time
            )
            #print(denoise_probs)
            return self.calculate_cross_entropy_from_probs(p_dist=q_revs, q_dist=denoise_probs)
        else:
            return self.cross_entropy_logits(logits, q_revs)
    

    def get_reconstruction_term_loss(self, batch, drop_condition):
        #get the known forward noising probabilites for going from x0 -> x1  
        time = torch.ones(size=(batch.batch_size,), dtype=torch.long)

        x_1 = self.noiser.noise_x0_xt(
            x0_batch=batch.x*1.0, 
            time_batch=torch.ones(size=(batch.batch_size,), dtype=torch.long)[batch.batch]
        )

        q_forward = self.noiser.get_transition_probabilities(
            x_t_batch=x_1*1.0,
            time_batch=time[batch.batch],
            scheduler=self.scheduler
        )

        logits = self.denoiser.get_logits(
            x_t=x_1*1.0,
            batch=batch,
            time=time,
            scheduler=self.scheduler,
            drop_cond=drop_condition   
        )

        #Apply the x0 reparameterization as outlined in D3PM if desired
        #Return the cross-entropy between the forward noising step and the predicted reversal.
        if self.use_x0_reparam:
            denoise_probs = self.perform_x0_reparam(
                denoise_logits=logits,
                x_t=x_1*1.0,
                batch=batch,
                time=time
            )
            return self.calculate_cross_entropy_from_probs(p_dist=q_forward, q_dist=denoise_probs)
        else:
            return self.cross_entropy_logits(logits, q_forward)
    
    def calculate_loss(self, batch, batch_idx):
        loss = 0.0
        #Determining whether conditioning should be dropped
        drop_condition = True if torch.rand(1) <= self.drop_prob else False
   
        loss+=self.get_denoise_matching_term_loss(
            batch=batch,
            drop_condition=drop_condition
        )
        #print(loss)

        #Calculating the reconstruction term and adding it to the total loss
        loss += self.get_reconstruction_term_loss(
            batch=batch,
            drop_condition=drop_condition
        )
       
        #If desired add the auxillary term as described in the D3PM paper
        if self.auxillary_weight is not None:
            aux_loss =  self.get_auxillary_term_loss(
                batch=batch,
                drop_condition=drop_condition
            )
            loss += self.auxillary_weight*aux_loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, batch_idx=batch_idx)
        self.log("train_loss", loss, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, batch_idx=batch_idx)
        self.log("val_loss", loss, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, batch_idx=batch_idx)
        self.log("test_loss", loss, on_epoch=True, batch_size=batch.batch_size)
        return loss
    
    def get_elem_from_one_hot(self, one_hot_vector:torch.tensor):
        indices = torch.argmax(one_hot_vector, dim=-1)
        return [self.element_pool[index] for index in indices]

    def sample(self, 
               n_samples:int, 
               conditionings:torch.tensor,
               template_atoms:Atoms,
               guidance_scale:float=2.0,
               return_as_atoms_list:bool=False, 
               batch_size:int=40,
               timesteps:torch.tensor=None,
               log_all_timesteps:bool=False
        ):
        noised_xs = [self.noiser.sample_from_stationary(num_atoms=len(template_atoms)) for _ in range(n_samples)]
        sample_dataset = self.denoiser.get_sample_dataset(
            noised_xs=noised_xs, 
            conditionings=conditionings,
            template_atoms=template_atoms
        )
        sample_loader = self.denoiser.get_sample_loader(
            dataset=sample_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        samples = []
        for batch in sample_loader:
            denoised_batch_list = self.denoise_batch(
                batch=batch, 
                guidance_scale=guidance_scale, 
                timesteps=timesteps,
                log_all_timesteps=log_all_timesteps
            )
            result_list = self.convert_denoised_batches_to_traj(
                denoised_batch_list=denoised_batch_list,
                batch_size=batch.batch_size,
                return_as_atoms_list=return_as_atoms_list
            )
            samples+=result_list
        return samples
    
    def denoise_batch(self, batch, guidance_scale, timesteps, log_all_timesteps):
        batch_list = []
        if timesteps is None:
            timesteps = torch.arange(self.scheduler.t_init, self.scheduler.t_final+1, 1).flip(dims=(0,))
        for timestep in timesteps:
            if log_all_timesteps:
                batch_list.append(batch.clone())
            else:
                batch_list = [batch.clone()]
            xs_denoised = self.single_denoise_step(
                batch=batch, 
                time=timestep, 
                guidance_scale=guidance_scale
            )
            batch.x = xs_denoised
        return batch_list

    def get_distribution(self, probabilites:torch.tensor) -> Categorical:
        return Categorical(probs=probabilites)

    def sample_onehot_vectors(self, probabilities:torch.tensor):
        distribution = self.get_distribution(probabilites=probabilities)
        samples = distribution.sample()
        onehots = F.one_hot(samples, num_classes=len(self.element_pool))
        return onehots

    def single_denoise_step(self, batch, time, guidance_scale:float=2.0):
        ts = time*torch.ones(size=(batch.batch_size,), dtype=torch.long)
        probs = self.get_reverse_transition_probabilities(
            x_t=batch.x*1.0,
            batch=batch, 
            time=ts, 
            guidance_scale=guidance_scale
        )
        if time == self.scheduler.t_init and self.denoiser.absorbing_state:
            probs[:,self.denoiser.absorbing_state_index] = 0.0
        xs_denoised = self.sample_onehot_vectors(probabilities=probs)
        return xs_denoised

    def convert_denoised_batches_to_traj(self, denoised_batch_list, batch_size, return_as_atoms_list:bool=False):
        num_timesteps = len(denoised_batch_list)#, num_samples, num_atoms, n_classes = denoised_xs_batched.shape
        result_list = []
        for sample_idx in range(batch_size):
            denoise_traj = []
            for timestep in range(num_timesteps):
                data = denoised_batch_list[timestep].get_example(sample_idx)
                if return_as_atoms_list:
                    sample = data.to_atoms(self.element_pool)
                else:
                    sample = data.to_elems(self.element_pool)
                denoise_traj.append(sample)
            result_list.append(denoise_traj)
        return result_list

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1, #Reducing factor
                patience=10 #Patience for scheduler
            ),
            'monitor': 'val_loss', #monitor the validation loss
            'interval': 'epoch',    #monitor at epoch level
            'frequency': 1              #with frequency 1
        }
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    @property
    def const_state_dict(self):
        const_state_dict = {
            "element_pool":self.element_pool,
            "random_seed":self.random_seed,
            "drop_prob":self.drop_prob,
            "lr":self.lr,
            "weight_decay":self.weight_decay,
            "use_x0_reparam":self.use_x0_reparam,
            "auxillary_weight":self.auxillary_weight
        }
        modules = {"scheduler_info":self.scheduler, "denoiser_info":self.denoiser, "noiser_info":self.noiser}
        for module_type in modules:
            const_state_dict[module_type] = modules[module_type].const_state_dict
        return const_state_dict

    def get_scheduler_from_checkpoint(self, scheduler_params):
        implemented_schedulers = {
            "LinearScheduler":LinearScheduler,
            "CosineScheduler":CosineScheduler,
            "ExponentialScheduler":ExponentialScheduler
        }
        scheduler_type = scheduler_params.pop("scheduler_type")
        if scheduler_type in implemented_schedulers:
            scheduler = implemented_schedulers[scheduler_type](**scheduler_params)
            return scheduler
        else:
            raise Exception(f"Scheduler of type: {scheduler_type} has not been implemented yet")

    def get_noiser_from_checkpoint(self, noiser_params, element_pool):
        implemented_noisers = {
            "AbsorbingStateNoiser":AbsorbingStateNoiser,
            "UniformTransitionsNoiser":UniformTransitionsNoiser
        }
        noiser_type = noiser_params.pop("noiser_type")
        if noiser_type in implemented_noisers:
            noiser = implemented_noisers[noiser_type](**noiser_params, element_pool=element_pool)
            return noiser
        else:
            raise Exception(f"Noiser of type: {noiser_type} has not been implemented yet")

    def get_denoiser_from_checkpoint(self, denoiser_params, element_pool):
        implemented_denoisers = {
            "DiscreteGNNDenoiser":DiscreteGNNDenoiser
        }
        denoiser_type = denoiser_params.pop("denoiser_type")
        if denoiser_type in implemented_denoisers:
            cond_embedder = self.get_condition_embedder_from_checkpoint(cond_embedder_params=denoiser_params.pop("condition_info"))
            denoiser = implemented_denoisers[denoiser_type](**denoiser_params, cond_embedder=cond_embedder, element_pool=element_pool)
            return denoiser
        else:
            raise Exception(f"Denoiser of type: {denoiser_type} has not been implemented yet")

    def get_condition_embedder_from_checkpoint(self, cond_embedder_params):
        implemented_embedders = {
            "RateEmbedder":RateEmbedder, 
            "ClassLabelEmbedder":ClassLabelEmbedder,
            "RateClassEmbedder":RateClassEmbedder
        }
        embedder_type = cond_embedder_params.pop("embedding_type")
        if embedder_type in implemented_embedders:
            cond_embedder = implemented_embedders[embedder_type](**cond_embedder_params)
            return cond_embedder
        else:
            raise Exception(f"Denoiser of type: {embedder_type} has not been implemented yet")


    def on_save_checkpoint(self, checkpoint):
        checkpoint["diffusion_parameters"] = self.const_state_dict
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        cfg = checkpoint["diffusion_parameters"]
        self.element_pool = cfg.pop("element_pool")
        self.random_seed = cfg.pop("random_seed")
        self.drop_prob = cfg.pop("drop_prob")
        self.weight_decay = cfg.pop("weight_decay")
        self.use_x0_reparam = cfg.pop("use_x0_reparam")
        self.auxillary_weight = cfg.pop("auxillary_weight")
        self.lr = cfg.pop("lr")
        self.scheduler = self.get_scheduler_from_checkpoint(scheduler_params=cfg.pop("scheduler_info"))
        self.noiser = self.get_noiser_from_checkpoint(noiser_params=cfg.pop("noiser_info"), element_pool=self.element_pool)
        self.noiser.pre_compute_accum_q_matrices(self.scheduler)
        self.denoiser = self.get_denoiser_from_checkpoint(denoiser_params=cfg.pop("denoiser_info"), element_pool=self.element_pool)
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)

