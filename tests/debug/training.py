from gen_catalyst_design.discrete_space_diffusion import (
    DiffusionModel, DiscreteGNNDenoiser, AbsorbingStateNoiser, UniformTransitionsNoiser,
    CosineScheduler, ClassLabelEmbedder
)

from gen_catalyst_design.utils import (
    setup_trainer_and_logger
)

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)



from ase.atoms import Atoms
from ase.io import read, write
import torch


def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    use_absorbing_state = True
    mask_classes = True
    element_pool = ["Au","Cu","Pd","Rh","Ni","Ga"]
    if use_absorbing_state:
        element_pool = ["(X)"] + element_pool

    atoms_list = read("dataset.traj", index=":")
    if mask_classes:
        for atoms in atoms_list:
            atoms.info["class"] = 0

    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=atoms_list,
        element_pool=element_pool,
        batch_size=50,
    )

    scheduler = CosineScheduler(beta_max=1e-1, beta_min=1e-4)
    noiser = AbsorbingStateNoiser(element_pool=element_pool)

    hidden_dim = 28

    conditioning = ClassLabelEmbedder(num_labels=3, embedding_dim=hidden_dim)    
    denoiser = DiscreteGNNDenoiser(
        element_pool=element_pool,
        cond_embedder=conditioning,
        message_dim=hidden_dim,
        n_hidden_layers=1,
        hidden_dim_rep=hidden_dim
    )

    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        denoiser=denoiser,
        drop_prob=0.2
    )
    
    for batch in train_loader:
        diff_model.calculate_loss(batch=batch, batch_idx=None)

    trainer_kwargs={
        "max_epochs":-1,
        "log_every_n_steps":50, 
        "enable_progress_bar":True, 
        "enable_model_summary":True
    }

    trainer = setup_trainer_and_logger(
        model_name="AbsorbingStateModel",
        patience=40,
        trainer_kwargs=trainer_kwargs
    )

    trainer.fit(
        model=diff_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )




if __name__ == "__main__":
    main()