from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from ase_ml_models.databases import get_atoms_list_from_db
from ase.db import connect
from ase.io import read, write
import torch
import os


def main():
    noiser_type = "Uniform" # Absorbing | Uniform
    miller_index = "100"
    guidance_scale = 1.2
    model_type = "last" # Best/epoch | last
    n_samples = 100
    model_num = "001"
    condition = 0

    pth_header = os.path.join(noiser_type, f"model_{model_num}", "checkpoints")
    checkpoint_files = os.listdir(pth_header)
    checkpoint_file = ""
    for file in checkpoint_files:
        if model_type in file:
            checkpoint_file = file
            break
    
    diff_model = DiffusionModel.load_from_checkpoint(os.path.join(pth_header, checkpoint_file))
    diff_model = diff_model.to(device=torch.device("cpu"))

    db = connect(f"../../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]

    result_samples = diff_model.sample(
        guidance_scale=guidance_scale,
        n_samples=n_samples, 
        conditionings=condition*torch.ones(size=(n_samples,), dtype=torch.long), 
        template_atoms=template_atoms, 
        batch_size=50, 
        timesteps=None, 
        log_all_timesteps=False, 
        return_as_atoms_list=True
    )

    atoms_list = [sample[0] for sample in result_samples]
    write(filename=f"{noiser_type}_{guidance_scale}_scale.traj", images=atoms_list)

if __name__ == "__main__":
    main()