from gen_catalyst_design.discrete_space_diffusion import (
    DiffusionModel, MPNNLogitPredictor, ExponentialScheduler, 
    CosineScheduler, LinearScheduler, AbsorbingStateNoiser,
    UniformTransitionsNoiser, RateConditioning
    )

from gen_catalyst_design.utils import (
    setup_trainer_and_logger,
    get_full_element_pool
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)

import random
import torch


def main():
    #General settings
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    element_pool = get_full_element_pool()
    drop_prob = 0.1
    hidden_dim = 32
    n_interaction_blocks = 5


    #Noiser settings
    noiser_type = "absorbing"

    #Scheduler settings
    scheduler_type = "exponential"

    scheduler_kwargs = {
        "beta_min":1e-4,
        "beta_max":5e-2,
        "time_sample_method":"stratified"
    }

    #Conditioning settings
    conditioning_types = ["rate"]


    #Construct diffusion model
    if noiser_type == "absorbing":
        element_pool = ["(X)"] + element_pool
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool
        )
    elif noiser_type == "uniform":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool
        )
    else:
        raise Exception(f"noiser of type {noiser_type} is not implemented")

    if scheduler_type == "exponential":
        scheduler = ExponentialScheduler(
            **scheduler_kwargs
        )
    elif scheduler_type == "cosine":
        scheduler_kwargs.update({"reg":0.1})
        scheduler = CosineScheduler(
            **scheduler_kwargs
        )
    elif scheduler_type == "linear":
        scheduler = LinearScheduler(
            **scheduler_kwargs
        )
    else:
        raise Exception(f"scheduler of type {scheduler_type} is not implemented")
    
    conditionings = []
    for cond_type in conditioning_types:
        if cond_type == "rate":
            conditioning = RateConditioning(
                embedding_dim=hidden_dim
            )
        if cond_type == "e_form":
            raise Exception("Not implemented yet")
        conditionings.append(conditioning)


    logit_predictor = MPNNLogitPredictor(
        num_elements=len(element_pool),
        embedding_dim=64,
        conditioning_dim=
    )

    DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        logit_predictor=logit_predictor,
        conditioning=
        drop_prob=drop_prob=
    )

if __name__ == "__main__":
    main()