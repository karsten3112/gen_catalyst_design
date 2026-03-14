from .schedulers import DiscreteTimeScheduler, ExponentialBetaScheduler, CosineScheduler, LinearBetaScheduler
from .noisers import DiscreteSpaceNoiser, UniformTransitionsNoiser, AbsorbingStateNoiser
from .logits import MPNNLogitPredictor, LogitPredictor, TransformerLogitPredictor
from .conditioning import Conditioning, NoneConditioning, RateConditioning
from torch_geometric.utils import to_dense_batch
from .guidance import ReactionRateModule, rateGNN
from ase.atoms import Atoms
from torch_geometric.utils import scatter
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ase.atoms import Atoms

implemented_modules = {
    "noiser":{
        "AbsorbingStateNoiser":AbsorbingStateNoiser,
        "UniformTransitionsNoiser":UniformTransitionsNoiser
    },
    "scheduler":{
        "ExponentialBetaScheduler":ExponentialBetaScheduler,
        "CosineScheduler":CosineScheduler,
        "LinearBetaScheduler":LinearBetaScheduler
    },
    "logit_predictor":{
        "MPNNLogitPredictor":MPNNLogitPredictor,
        "TransformerLogitPredictor":TransformerLogitPredictor
    },
    "conditioning":{
        "None":NoneConditioning,
        "RateConditioning":RateConditioning
    }

}

class DiffusionModel(LightningModule):
    def __init__(
            self,
            element_pool:list=[],
            scheduler:DiscreteTimeScheduler=None,
            noiser:DiscreteSpaceNoiser=None,
            logit_predictor:LogitPredictor=None,
            rate_conditioning:Conditioning=NoneConditioning(),
            e_form_conditining:Conditioning=NoneConditioning(),
            drop_prob:float=0.1,
            lr:float=1e-3,
            weight_decay:float=0.0,
            num_kl_div_estimates:int=5,
            use_x0_reparam:bool=True,
            d3pm_auxillary_weight:float=None,
            auxillary_rate_weight:float=None,
            log_regularization:float=1e-12,
        ):
        super().__init__()
        self.element_pool = element_pool
        self.scheduler = scheduler
        self.noiser = noiser
        self.logit_predictor = logit_predictor
        self.rate_conditioning = rate_conditioning
        self.e_form_conditioning = e_form_conditining

        if self.noiser is not None and self.noiser.accumulated_q_matrices is None:
            self.noiser.pre_compute_accum_q_matrices(scheduler=self.scheduler)
        
        self.drop_prob = drop_prob
        self.drop_pattern_matrix = [
            [False, False], #Drop no condition
            [False, True],  #Drop only e_form
            [True, False],  #Drop only rate
            [True, True]    #Drop both conditions
            ]
        self.drop_prob_vector = self.get_drop_prob_vector()
        self.drop_scenarios_dict = {
            "uncond": self.drop_pattern_matrix[-1],
            "rate": self.drop_pattern_matrix[1],
            "e_form":self.drop_pattern_matrix[2]
        }
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_kl_div_estimates = num_kl_div_estimates
        self.use_x0_reparam = use_x0_reparam
        self.d3pm_auxillary_weight = d3pm_auxillary_weight
        self.auxillary_rate_weight = auxillary_rate_weight
        self.log_regularization = log_regularization
        self.cross_entropy_logits = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def get_drop_prob_vector(self):
        if type(self.rate_conditioning) != NoneConditioning and type(self.e_form_conditioning) != NoneConditioning:
            keep_rate_prob = (1.0 - self.drop_prob)/4.0
            keep_e_form_prob = keep_rate_prob
        elif type(self.rate_conditioning) == NoneConditioning and type(self.e_form_conditioning) != NoneConditioning:
            keep_rate_prob = 0.0
            keep_e_form_prob = 2.0*(1.0 - self.drop_prob)/4.0
        elif type(self.rate_conditioning) != NoneConditioning and type(self.e_form_conditioning) == NoneConditioning:
            keep_rate_prob = 2.0*(1.0 - self.drop_prob)/4.0
            keep_e_form_prob = 0.0
        else:
            keep_rate_prob = 0.0
            keep_e_form_prob = 0.0
            
        drop_prob_vector = torch.tensor([
            keep_rate_prob+keep_e_form_prob, #prob. for kepping both conditionings
            keep_rate_prob, #prob. for keeping rate-condition alone
            keep_e_form_prob, #prob. for keeping e_form-condition alone
            self.drop_prob] #prob. for dropping both conditinings
        ) 
        #if drop_prob_matrix.sum() != 1.0:
        #    raise Exception(f"matrix entries are not normalized {drop_prob_matrix} perhaps set p_drop to zero when no condition is available")
        return drop_prob_vector

    def on_sample_start(self):
        device = self.device
        self.noiser.set_device(device=device)
        self.scheduler.set_device(device=device)
        self.logit_predictor.set_device(device=device)
        self.rate_conditioning.set_device(device=device)
        self.e_form_conditioning.set_device(device=device)

    def on_fit_start(self):
        self.train()
        device = self.device
        self.noiser.set_device(device=device)
        self.scheduler.set_device(device=device)
        self.logit_predictor.set_device(device=device)
        self.rate_conditioning.set_device(device=device)
        self.e_form_conditioning.set_device(device=device)

    def perform_x0_reparam(
            self, 
            denoise_logits, 
            x_t, 
            batch,
            time
        ):
        denoise_probs = self.logit_predictor.get_probs_from_logits(
            logits=denoise_logits
        )
        x0s = [
            F.one_hot(torch.tensor(i, device=self.device), num_classes=len(self.element_pool)) * \
            torch.ones(size=(len(x_t), 1), device=self.device) 
            for i in range(len(self.element_pool))
        ]
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
    
    def get_kl_divergence(self, p_dist, q_dist, batch):
        p_q_cross_entropy = self.get_cross_entropy(p_dist=p_dist, q_dist=q_dist, batch=batch)
        p_entropy = self.get_cross_entropy(p_dist=p_dist, q_dist=p_dist, batch=batch)
        return p_q_cross_entropy-p_entropy

    def get_cross_entropy(self, p_dist, q_dist, batch):
        cross_entropy = -(p_dist*torch.log(q_dist+self.log_regularization)).sum(dim=-1)
        graph_cross_entropy = scatter(cross_entropy, batch.batch, dim=0, reduce="mean")
        return graph_cross_entropy


    def get_denoise_matching_term_loss(self, noised_batch, x0_batch, time, embedded_condition):
        #Calculate the known true posterier for when x0 is known
        loss = 0.0
        q_revs = self.noiser.get_reverse_transition_probabilities(
            x0_batch=x0_batch.x*1.0,
            x_t_batch=noised_batch.x*1.0, 
            time_batch=time[noised_batch.batch], 
            scheduler=self.scheduler
        )
        q_revs = torch.clamp(q_revs, min=1e-7, max=1.0 - 1e-7)
        logits = self.logit_predictor.get_logits(
            batch=noised_batch,
            time=time,
            embedded_condition=embedded_condition   
        )
        #Apply the x0 reparameterization as outlined in D3PM if desired
        #return the cross entropy between the true posterier and the predicted reversals
        #Note here that this is equal to the KL-divergence up to a constant which is not learnable
        if self.use_x0_reparam:
            denoise_probs = self.perform_x0_reparam(
                denoise_logits=logits,
                x_t=noised_batch.x*1.0,
                batch=noised_batch,
                time=time
            )
            kl_div_per_graph = self.get_kl_divergence(p_dist=q_revs, q_dist=denoise_probs, batch=noised_batch)
            loss += kl_div_per_graph.mean()
        else:
            loss += self.cross_entropy_logits(logits, q_revs)
        return loss


    def get_d3pm_auxillary_term_loss(self, noised_batch, x0_batch, time, embedded_condition):
        #Calculate the auxillary term as mentioned in D3PM paper
        logits = self.logit_predictor.get_logits(
            batch=noised_batch,
            time=time,
            embedded_condition=embedded_condition 
        )

        probs = self.logit_predictor.get_probs_from_logits(
            logits=logits
        )

        q_forward = self.noiser.get_accum_transition_probabilities(
            x0_batch=x0_batch.x*1.0,
            time_batch=time[noised_batch.batch]
        )
        return self.get_cross_entropy(p_dist=q_forward, q_dist=probs, batch=noised_batch).mean()


    def get_reconstruction_term_loss(self, batch, embedded_condition):
        #get the known forward noising probabilites for going from x0 -> x1  
        batch_1 = batch.clone()
        time = torch.ones(size=(batch.batch_size,), dtype=torch.long, device=self.device)

        self.noiser.noise_batch_x0_xt(
            batch=batch_1,
            time_batch=time[batch.batch]
        )

        q_forward = self.noiser.get_accum_transition_probabilities(
            x0_batch=batch.x*1.0,
            time_batch=time[batch.batch]
        )

        logits = self.logit_predictor.get_logits(
            batch=batch_1,
            time=time,
            embedded_condition=embedded_condition  
        )
        
        #Apply the x0 reparameterization as outlined in D3PM if desired
        #Return the cross-entropy between the forward noising step and the predicted reversal.
        if self.use_x0_reparam:
            denoise_probs = self.perform_x0_reparam(
                denoise_logits=logits,
                x_t=batch_1.x*1.0,
                batch=batch,
                time=time
            )
            return self.get_cross_entropy(p_dist=batch.x*1.0, q_dist=denoise_probs, batch=batch).mean()#self.get_kl_divergence(p_dist=q_forward, q_dist=denoise_probs, batch=batch_1).mean()
        else:
            return self.cross_entropy_logits(logits, q_forward)

    def get_auxillary_rate_loss(self, batch, time, embedded_condition=None):
        x0_rates = self.logit_predictor.get_x0_rate_prediction(
            batch=batch,
            time=time[batch.batch],
            embedded_condition=self.conditioning.get_condition_embedding(
                condition=batch.y,
                batch_size=batch.batch_size,
                drop_condition=True
            )[batch.batch]
        )
        return F.mse_loss(x0_rates.squeeze(-1), torch.log(batch.y))

    def get_joint_embedded_condition(self, batch, drop_scenario):
        resulting_condition = 0.0
        not_none_condition = False
        for drop_condition, embedder, condition in zip(
                drop_scenario, 
                [self.rate_conditioning, self.e_form_conditioning], 
                [batch.rate if "rate" in batch else None, batch.e_form if "e_form" in batch else None]
            ):
            embedded_condition = embedder.get_condition_embedding(
                condition=condition,
                batch_size=batch.batch_size,
                drop_condition=drop_condition
            )
            if embedded_condition is not None:
                resulting_condition += embedded_condition
                not_none_condition = True
        if not_none_condition:
            return resulting_condition
        else:
            return None
            
    def calculate_loss_terms(self, batch, batch_idx):
        idx = torch.multinomial(self.drop_prob_vector, 1).squeeze(-1)
        drop_scenario = self.drop_pattern_matrix[idx]
        embedded_condition = self.get_joint_embedded_condition(batch=batch, drop_scenario=drop_scenario)
        #drop_condition = True if torch.rand(1) <= self.drop_prob else False
        #embedded_condition = self.rate_conditioning.get_condition_embedding(
        #        condition=batch.rate,
        #        batch_size=batch.batch_size,
        #        drop_condition=drop_condition
        #    )
        
        if embedded_condition is not None:
            embedded_condition = embedded_condition[batch.batch]

        t_span = (2, self.scheduler.t_final)
        denoise_matching_term = 0.0
        aux_loss = 0.0
        aux_rate_loss = 0.0
        for _ in range(self.num_kl_div_estimates):
            batch_t = batch.clone()
            time = self.scheduler.sample_time(n_samples=batch.batch_size, t_span=t_span)
            self.noiser.noise_batch_x0_xt(
                batch=batch_t,
                time_batch=time[batch.batch]
            )
            denoise_matching_term += (t_span[1]-t_span[0])*self.get_denoise_matching_term_loss(
                noised_batch=batch_t,
                x0_batch=batch,
                time=time,
                embedded_condition=embedded_condition
            )
            #If desired add the auxillary term as described in the D3PM paper
            if self.d3pm_auxillary_weight is not None:
                aux_loss += self.d3pm_auxillary_weight*self.get_d3pm_auxillary_term_loss(
                    noised_batch=batch_t,
                    x0_batch=batch,
                    time=time,
                    embedded_condition=embedded_condition
                )
            #If desired add the auxillary-rate loss
            if self.auxillary_rate_weight is not None:
                aux_rate_loss += self.auxillary_rate_weight*self.get_auxillary_rate_loss(
                    batch=batch_t,
                    time=time,
                    embedded_condition=embedded_condition,
                )
        
        denoise_matching_term/=self.num_kl_div_estimates
        aux_loss/=self.num_kl_div_estimates
        aux_rate_loss/=self.num_kl_div_estimates

        #Calculating the reconstruction term and adding it to the total loss
        reconstruction_term = self.get_reconstruction_term_loss(
            batch=batch,
            embedded_condition=embedded_condition
        )
        return denoise_matching_term, reconstruction_term, aux_loss, aux_rate_loss


    def training_step(self, batch, batch_idx):
        denoise_term, recon_term, aux_loss, aux_rate_loss = self.calculate_loss_terms(batch=batch, batch_idx=batch_idx)
        loss = denoise_term + recon_term + aux_loss + aux_rate_loss
        self.log("train_loss", loss, on_epoch=True, batch_size=batch.batch_size)
        self.log("train_loss/denoise", denoise_term, on_epoch=True, batch_size=batch.batch_size)
        self.log("train_loss/recon", recon_term, on_epoch=True, batch_size=batch.batch_size)
        self.log("train_loss/d3pm_aux_term", aux_loss, on_epoch=True, batch_size=batch.batch_size)
        self.log("train_loss/aux_rate_term", aux_rate_loss, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        denoise_term, recon_term, aux_loss, aux_rate_loss = self.calculate_loss_terms(batch=batch, batch_idx=batch_idx)
        loss = denoise_term + recon_term + aux_loss + aux_rate_loss
        self.log("val_loss", loss, on_epoch=True, batch_size=batch.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        denoise_term, recon_term, aux_loss, aux_elem_loss = self.calculate_loss_terms(batch=batch, batch_idx=batch_idx)
        loss = denoise_term + recon_term + aux_loss + aux_elem_loss
        self.log("test_loss", loss, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def configure_optimizers(self):
        params = [p for _, p in sorted(self.named_parameters(), key=lambda x: x[0])]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
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
        return {"optimizer":optimizer}#, "lr_scheduler":scheduler}


    def sample(
            self, 
            n_samples:int,
            template_atoms:Atoms, 
            conditioning_dicts:dict={},
            condition_keys:list=["rate"],
            guidance_scale_dict:dict={"rate":2.0},
            return_as_atoms_list:bool=False, 
            batch_size:int=40,
            timesteps:torch.tensor=None,
            log_all_timesteps:bool=False,
            temp:float=1.0,
            dataset_kwargs:dict={}
        ):
        self.eval()
        self.on_sample_start()
        noised_atoms = self.noiser.sample_atoms_from_stationary(
            n_samples=n_samples, 
            template_atoms=template_atoms
        )
        for atoms, condition_dict in zip(noised_atoms, conditioning_dicts):
            atoms.info.update(condition_dict)
        
        sample_loader = self.logit_predictor.get_sample_loader(
            atoms_list=noised_atoms,
            element_pool=self.element_pool,
            batch_size=batch_size,
            condition_keys=condition_keys,
            dataset_kwargs=dataset_kwargs
        )
        samples = []
        for batch in sample_loader:
            denoised_batch_list = self.denoise_batch(
                batch=batch, 
                guidance_scale_dict=guidance_scale_dict, 
                timesteps=timesteps,
                log_all_timesteps=log_all_timesteps,
                temp=temp
            )
            result_list = self.convert_denoised_batches_to_traj(
                denoised_batch_list=denoised_batch_list,
                batch_size=batch.batch_size,
                return_as_atoms_list=return_as_atoms_list
            )
            samples+=result_list
        return samples

    def denoise_batch(
            self, 
            batch, 
            guidance_scale_dict, 
            timesteps, 
            log_all_timesteps, 
            temp:float=1.0
        ):
        batch_list = []
        if timesteps is None:
            timesteps = torch.arange(self.scheduler.t_init, self.scheduler.t_final+1, 1, device=self.device).flip(dims=(0,))
        for timestep in timesteps:
            if log_all_timesteps:
                batch_list.append(batch.clone())
            else:
                batch_list = [batch.clone()]
            self.single_denoise_step(
                batch=batch, 
                time=timestep, 
                guidance_scale_dict=guidance_scale_dict,
                temp=temp
            )
        return batch_list
    
    def single_denoise_step(
            self, 
            batch, 
            time, 
            guidance_scale_dict:dict={"rate":2.0},
            masking_sites:torch.tensor=None, 
            temp:float=1.0
        ):
        ts = time*torch.ones(size=(batch.batch_size,), dtype=torch.long, device=self.device)
        probs = self.get_reverse_transition_probabilities(
            batch=batch, 
            time=ts, 
            guidance_scale_dict=guidance_scale_dict,
            temp=temp
        )
        if time == self.scheduler.t_init and self.noiser.absorbing_state:
                probs[:,self.noiser.absorbing_state_index] = 0.0
                probs/=probs.sum(dim=1, keepdim=True)

        if masking_sites is not None:
            xs_denoised = batch.x
            denoised_sites = self.noiser.sample_transition(probabilities=probs[masking_sites])
            xs_denoised[masking_sites] = denoised_sites
        else:
            xs_denoised = self.noiser.sample_transition(probabilities=probs)
        batch.x = xs_denoised

    def get_reverse_transition_probabilities(self, batch, time, guidance_scale_dict:dict={"rate":2.0}, temp:float=1.0):
        guided_logits = self.get_guided_logits(
            batch=batch,
            time=time,
            guidance_scale_dict=guidance_scale_dict,
            temp=temp
        )
        if self.use_x0_reparam:
            guided_probs = self.perform_x0_reparam(
                denoise_logits=guided_logits,
                x_t=batch.x*1.0,
                batch=batch,
                time=time
            )
            return guided_probs
        else:
            return self.logit_predictor.get_probs_from_logits(logits=guided_logits)

    def get_guided_logits(self, batch, time, guidance_scale_dict:dict={"rate":2.0}, temp:float=1.0):
        guided_logits_dict = {}
        for kw in ["uncond"]+list(guidance_scale_dict.keys()):
            embedded_condition = self.get_joint_embedded_condition(
                batch=batch,
                drop_scenario=self.drop_scenarios_dict[kw]
            )
            if embedded_condition is not None:
                embedded_condition = embedded_condition[batch.batch]
            logits = self.logit_predictor.get_logits(
                batch=batch,
                time=time,
                embedded_condition=embedded_condition
            )
            guided_logits_dict[kw] = logits
        guided_logits = guided_logits_dict["uncond"]
        for kw in guidance_scale_dict:
            guidance_scale = guidance_scale_dict[kw]
            guided_logits += guidance_scale*(guided_logits_dict[kw]-guided_logits_dict["uncond"])
        return guided_logits/temp

    def convert_denoised_batches_to_traj(self, denoised_batch_list, batch_size, return_as_atoms_list:bool=False):
        num_timesteps = len(denoised_batch_list)
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
    
    def gibbs_sample(
            self,
            n_samples:int,
            template_atoms:Atoms,
            block_iterations:int=1,
            block_size:int=4,
            conditioning_dicts:dict={},
            condition_key:str=None,
            guidance_scale:float=2.0,
            return_as_atoms_list:bool=False, 
            batch_size:int=40,
            timesteps:torch.tensor=None,
            log_all_timesteps:bool=False,
            temp:float=1.0,
            dataset_kwargs:dict={}
        ):
        noised_atoms = self.noiser.sample_atoms_from_stationary(
            n_samples=n_samples, 
            template_atoms=template_atoms
        )
        for atoms, condition_dict in zip(noised_atoms, conditioning_dicts):
            atoms.info.update(condition_dict)
        
        sample_loader = self.logit_predictor.get_sample_loader(
            atoms_list=noised_atoms,
            element_pool=self.element_pool,
            batch_size=batch_size,
            condition_key=condition_key,
            dataset_kwargs=dataset_kwargs
        )
        samples = []
        for batch in sample_loader:
            denoised_batch_list = self.denoise_batch_gibbs(
                batch=batch,
                guidance_scale=guidance_scale,
                block_iterations=block_iterations,
                block_size=block_size,
                timesteps=timesteps,
                log_all_timesteps=log_all_timesteps,
                temp=temp
            )
            result_list = self.convert_denoised_batches_to_traj(
                denoised_batch_list=denoised_batch_list,
                batch_size=batch.batch_size,
                return_as_atoms_list=return_as_atoms_list
            )
            samples+=result_list
        return samples

    def denoise_batch_gibbs(
            self,
            batch, 
            guidance_scale,
            block_iterations,
            block_size, 
            timesteps, 
            log_all_timesteps, 
            temp:float=1.0
        ):
        batch_list = []
        if timesteps is None:
            timesteps = torch.arange(self.scheduler.t_init, self.scheduler.t_final+1, 1).flip(dims=(0,))
        for timestep in timesteps:
            if log_all_timesteps:
                batch_list.append(batch.clone())
            else:
                batch_list = [batch.clone()]
            for _ in range(block_iterations):
                if timestep >= 10:
                    masking_sites = torch.hstack([torch.randperm(21)[:block_size]+i*21 for i in range(batch.batch_size)])
                else:
                    masking_sites = None
                self.single_denoise_step(
                    batch=batch, 
                    time=timestep, 
                    guidance_scale=guidance_scale,
                    masking_sites=masking_sites,
                    temp=temp
                )
        return batch_list
    

    def get_state_dict(self):
        state_dict = {
            "element_pool":self.element_pool,
            "drop_prob":self.drop_prob,
            "lr":self.lr,
            "weight_decay":self.weight_decay,
            "use_x0_reparam":self.use_x0_reparam,
            "d3pm_auxillary_weight":self.d3pm_auxillary_weight,
            "auxillary_rate_weight":self.auxillary_rate_weight,
            "num_kl_div_estimates":self.num_kl_div_estimates,
            "log_regularization": self.log_regularization
        }
        modules = {
            "scheduler_info":self.scheduler,
            "noiser_info":self.noiser,
            "logit_predictor_info":self.logit_predictor, 
            "rate_conditioning_info":self.rate_conditioning,
            "e_form_conditioning_info":self.e_form_conditioning
        }
        for module_type in modules:
            state_dict[module_type] = modules[module_type].get_state_dict()
        return state_dict
    
    def load_modules_from_checkpoint(self, parameter_dict):
        module_dict = {
            "scheduler":"scheduler",
            "noiser":"noiser",
            "logit_predictor":"logit_predictor",
            "rate_conditioning":"conditioning",
            "e_form_conditioning":"conditioning"
        }
        #module_list = ["scheduler", "noiser", "logit_predictor"]
        modules = {}
        for module_type in module_dict:
            info_key = module_type+"_info"
            if info_key not in parameter_dict:
                raise Exception(f"Module of type: {module_type} is not found in parameter dict, being {parameter_dict}")
            else:
                module_key = module_dict[module_type]
                module_params = parameter_dict[info_key]
                module_instance = module_params.pop(f"{module_key}_type")
                if module_type == "noiser":
                    modules[module_type] = implemented_modules[module_key][module_instance](element_pool=self.element_pool, **module_params)
                else:
                    modules[module_type] = implemented_modules[module_key][module_instance](**module_params)
        
        return modules
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["diffusion_parameters"] = self.get_state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        cfg = checkpoint["diffusion_parameters"]
        self.element_pool = cfg.pop("element_pool")
        self.drop_prob = cfg.pop("drop_prob")
        self.weight_decay = cfg.pop("weight_decay")
        self.use_x0_reparam = cfg.pop("use_x0_reparam")
        self.d3pm_auxillary_weight = cfg.pop("d3pm_auxillary_weight")
        self.auxillary_rate_weight = cfg.pop("auxillary_rate_weight")
        self.num_kl_div_estimates = cfg.pop("num_kl_div_estimates")
        self.log_regularization = cfg.pop("log_regularization")
        self.lr = cfg.pop("lr")
        modules = self.load_modules_from_checkpoint(parameter_dict=cfg)
        self.scheduler = modules.pop("scheduler")
        self.noiser = modules.pop("noiser")
        if self.noiser.accumulated_q_matrices is None:
            self.noiser.pre_compute_accum_q_matrices(self.scheduler)
        self.logit_predictor = modules.pop("logit_predictor")
        self.rate_conditioning = modules.pop("rate_conditioning")
        self.e_form_conditioning = modules.pop("e_form_conditioning")
        self.drop_prob_vector = self.get_drop_prob_vector()
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)