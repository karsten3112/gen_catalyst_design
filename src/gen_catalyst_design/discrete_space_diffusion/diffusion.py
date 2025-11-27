from .schedulers import DiscreteTimeScheduler, ExponentialScheduler, CosineScheduler, LinearScheduler
from .noisers import DiscreteSpaceNoiser, UniformTransitionsNoiser, AbsorbingStateNoiser
from .denoisers import DiscreteSpaceDenoiser, DiscreteGNNDenoiser
from .conditioning import ConditioningEmbedder, RateEmbedder, ClassLabelEmbedder
from ase.atoms import Atoms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ase.atoms import Atoms


class DiffusionModel(LightningModule):
    def __init__(
            self,
            element_pool:list=[],
            scheduler:DiscreteTimeScheduler=None,
            noiser:DiscreteSpaceNoiser=None,
            denoiser:DiscreteSpaceDenoiser=None,
            drop_prob:float=0.2,
            random_seed:int=42,
            loss_fn:callable=nn.CrossEntropyLoss(),
            lr:float=1e-4,
            weight_decay:float=1e-4,
        ):
        super().__init__()
        self.element_pool = element_pool
        self.scheduler = scheduler
        self.noiser = noiser
        self.denoiser = denoiser
        self.random_seed = random_seed
        self.drop_prob = drop_prob
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        if self.noiser is not None:
            if self.noiser.accumulated_q_matrices is None:
                self.noiser.pre_compute_accum_q_matrices(scheduler=self.scheduler)


    def on_fit_start(self):
        device = self.device
        self.noiser.set_device(device=device)
        self.scheduler.set_device(device=device)

    def calculate_loss(self, batch, batch_idx):
        time = self.scheduler.sample_time(n_samples=batch.batch_size)
        x_t = self.noiser.noise_x0_xt(x0_batch=batch.x*1.0, time_batch=time[batch.batch])
        drop_condition = True if torch.rand(1) <= self.drop_prob else False
        denoise_logits = self.denoiser.forward(
            x_t=x_t*1.0, 
            batch=batch, 
            time=time, 
            scheduler=self.scheduler, 
            drop_condition=drop_condition
        )
        q_revs = self.noiser.get_reverse_transition_probabilities(
            x0_batch=batch.x*1.0,
            x_t_batch=x_t*1.0, 
            time_batch=time[batch.batch], 
            scheduler=self.scheduler
        )
        loss = self.loss_fn(denoise_logits, q_revs)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, batch_idx=batch_idx)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, batch_idx=batch_idx)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, batch_idx=batch_idx)
        self.log("test_loss", loss, on_epoch=True)
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
            if log_all_timesteps:
                denoised_xs_batched = [batch.x.reshape(batch.batch_size, len(template_atoms), len(self.element_pool))]
            else:
                denoised_xs_batched = []
            
            denoised_batch_list = self.denoiser.denoise_batch(
                batch=batch, 
                scheduler=self.scheduler, 
                guidance_scale=guidance_scale, 
                timesteps=timesteps,
                log_all_timesteps=log_all_timesteps
            )
            denoised_xs_batched = torch.stack(
                denoised_xs_batched +
                [denoised_batch.reshape(batch.batch_size, len(template_atoms), len(self.element_pool)) for denoised_batch in denoised_batch_list]
            )
            
            result_list = self.convert_denoised_x_batch_to_traj(
                denoised_xs_batched=denoised_xs_batched, 
                template_atoms=template_atoms, 
                return_as_atoms_list=return_as_atoms_list
            )
            samples+=result_list
        return samples
    
    def convert_sample_dict_to_atoms(self, sample_dict, template_atoms):
        atoms = template_atoms.copy()
        updated_elements = ["O" if elem == "(X)" else elem for elem in sample_dict["elements"]]
        atoms.symbols = updated_elements
        atoms.info["sample_num"] = sample_dict["sample_num"]
        atoms.info["timestep"] = sample_dict["timestep"]
        return atoms

    def convert_denoised_x_batch_to_traj(self, denoised_xs_batched:torch.tensor, template_atoms:Atoms, return_as_atoms_list:bool=False):
        num_timesteps, num_samples, num_atoms, n_classes = denoised_xs_batched.shape
        result_list = []
        for sample in range(num_samples):
            denoising_traj = denoised_xs_batched[:,sample,:]
            denoise_list = []
            for timestep in range(num_timesteps):
                elements = self.get_elem_from_one_hot(one_hot_vector=denoising_traj[timestep])
                denoise_list.append({"elements":elements, "timestep":num_timesteps-timestep, "sample_num":sample+1})
            if return_as_atoms_list:
                atoms_list = [self.convert_sample_dict_to_atoms(sample_dict=sample_dict, template_atoms=template_atoms) for sample_dict in denoise_list]
                result_list.append(atoms_list)
            else:
                result_list.append(denoise_list)
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
            "weight_decay":self.weight_decay
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
            "ClassLabelEmbedder":ClassLabelEmbedder
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
        self.lr = cfg.pop("lr")
        self.scheduler = self.get_scheduler_from_checkpoint(scheduler_params=cfg.pop("scheduler_info"))
        self.noiser = self.get_noiser_from_checkpoint(noiser_params=cfg.pop("noiser_info"), element_pool=self.element_pool)
        self.denoiser = self.get_denoiser_from_checkpoint(denoiser_params=cfg.pop("denoiser_info"), element_pool=self.element_pool)
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)

