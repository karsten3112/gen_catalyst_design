from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from ase_ml_models.databases import get_atoms_list_from_db
from ase.db import connect
from ase.io import read, write
import torch
import os
import random


def main():
    random_seed = 42
    n_samples = 100
    miller_index = "100"
    ckpt_file_type = "last"
    models = [
        #"model_001",
        #"model_002",
        #"model_003",
        #"model_004"
        "model_009"
    ]
    
    cls = 22.0

    db = connect(f"../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]
    #template_atoms = read("../isolated_active_sites.traj", index=0)
    for model in models:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        
        ckpt_dir = os.path.join(model, "checkpoints")
        files = os.listdir(ckpt_dir)
        for file in files:
            if ckpt_file_type in file:
                ckpt_file = file

        diff_model = DiffusionModel.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file))
        diff_model = diff_model.to(device=torch.device("cpu"))
        conditioning_dicts = [{"rate": cls} for _ in range(n_samples)]
        result_samples = diff_model.sample(
            guidance_scale=10.0,
            conditioning_dicts=conditioning_dicts,
            n_samples=n_samples, 
            condition_key="rate",
            template_atoms=template_atoms, 
            batch_size=50, 
            timesteps=None, 
            log_all_timesteps=False, 
            return_as_atoms_list=True,
            temp=0.5
        )
        atoms_list = [sample[0] for sample in result_samples]
        write(filename=os.path.join(model, f"samples_{cls}.traj"), images=atoms_list)

if __name__ == "__main__":
    main()