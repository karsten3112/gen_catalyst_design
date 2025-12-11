from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from ase_ml_models.databases import get_atoms_list_from_db
from ase.db import connect
from ase.io import read, write
import torch
import os
import random


def main():
    random_seed = 42
    noiser_types = ["Absorbing"] # Absorbing | Uniform
    guidance_scales = [0.8, 1.2, 2.0]
    n_samples = 100
    miller_index = "100"
    model_type = "epoch" # Best/epoch | last
    condition = 0

    db = connect(f"../../../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]

    for noiser_type in noiser_types:
        out_dir = os.path.join(noiser_type, "samples")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        pth_header = os.path.join(noiser_type, "checkpoints")
        checkpoint_files = os.listdir(pth_header)
        checkpoint_file = ""
        for file in checkpoint_files:
            if model_type in file:
                checkpoint_file = file
                break 
        
        diff_model = DiffusionModel.load_from_checkpoint(os.path.join(pth_header, checkpoint_file))
        diff_model = diff_model.to(device=torch.device("cpu"))

        for guidance_scale in guidance_scales:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
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
            write(filename=os.path.join(out_dir, f"g_{guidance_scale}_scale.traj"), images=atoms_list)

if __name__ == "__main__":
    main()