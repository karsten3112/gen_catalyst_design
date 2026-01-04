
import random
import torch
from gen_catalyst_design.discrete_space_diffusion import (
    AbsorbingStateNoiser, UniformTransitionsNoiser, ExponentialScheduler, ClassLabelEmbedder,
    DiscreteGNNDenoiser, DiffusionModel
)
from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)
from ase.io import read

def main():
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    noiser_type = "Absorbing" # Absorbing | Uniform
    mark_active_sites = True
    use_edge_attr = False
    element_pool = ["Rh", "Cu", "Au", "Pd"]
    hidden_dim = 6
    
    if "Absorbing" in noiser_type:
        element_pool = ["(X)"] + element_pool

    if noiser_type == "Absorbing":
        noiser = AbsorbingStateNoiser(
            element_pool=element_pool
        )
    elif noiser_type == "Uniform":
        noiser = UniformTransitionsNoiser(
            element_pool=element_pool
        )
    else:
        raise Exception(f"noiser of type {noiser_type} is not implemented")

    scheduler = ExponentialScheduler(beta_max=1e-1, beta_min=1e-4)

    conditioning = ClassLabelEmbedder(num_labels=1, embedding_dim=hidden_dim)    
    denoiser = DiscreteGNNDenoiser(
        element_pool=element_pool,
        cond_embedder=conditioning,
        message_dim=hidden_dim,
        n_hidden_layers=0,
        hidden_dim_rep=hidden_dim,
        time_embedding_dim=hidden_dim,
        use_edge_attr=mark_active_sites,
        mark_active_sites=use_edge_attr,
        aggr="sum"
    )

    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        denoiser=denoiser,
        drop_prob=0.10,
        use_x0_reparam=True,
        auxillary_weight=0.0
    )

    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list = read("../training_set.traj", index=":"),
        element_pool=diff_model.element_pool,
        batch_size=1,
        do_train_shuffling=False,
        do_initial_shuffling=False
    )
    for batch in train_loader:
        time = torch.ones(size=(batch.batch_size,))
        final_reps = diff_model.denoiser.forward(
            x_t=batch.x*1.0,
            batch=batch,
            time=time,
            drop_condition=False,
        )
        probs = torch.softmax(final_reps, dim=-1)
        print(final_reps)
        break



if __name__ == "__main__":
    main()