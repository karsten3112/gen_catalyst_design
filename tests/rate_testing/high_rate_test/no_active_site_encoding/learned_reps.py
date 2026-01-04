from ase.db import connect
from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
import os
import torch
from ase_ml_models.databases import get_atoms_list_from_db
from ase.io import read, write

from gen_catalyst_design.discrete_space_diffusion.Dataset import (
    get_dataloaders_from_atoms_list
)

def main():
    pth_header = "Uniform/checkpoints"
    checkpoint_file = "last-v2.ckpt"


    diff_model = DiffusionModel.load_from_checkpoint(os.path.join(pth_header, checkpoint_file))
    diff_model = diff_model.to(device=torch.device("cpu"))

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
        #print(final_reps)
        probs = torch.softmax(final_reps, dim=-1)
        print(probs)
        break


if __name__ == "__main__":
    main()