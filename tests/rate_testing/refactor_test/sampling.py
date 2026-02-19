from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.discrete_space_diffusion.Dataset import add_site_connections
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
        #"class_0",
        "5mpl_no_aux_no_duplicates"
    ]
    temps = [0.5,1.0]#[0.2,0.5, 1.0]

    db = connect(f"../../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]
    dataset_kwargs = {
        "add_active_site_connectivity":False,
        "use_fully_connected_graph":False
    }

    for model in models:
        ckpt_dir = os.path.join(model, "checkpoints")
        files = os.listdir(ckpt_dir)
        for file in files:
            if ckpt_file_type in file:
                ckpt_file = file
        diff_model = DiffusionModel.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file))
        diff_model = diff_model.to(device=torch.device("cpu"))
        diff_model.noiser.label_smoothing = 0.0
        for temp in temps:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            result_samples = diff_model.sample(
                conditioning_dicts=[{"rate":8.0} for _ in range(n_samples)],
                guidance_scale=2.0,
                n_samples=n_samples, 
                condition_key="rate",
                template_atoms=template_atoms, 
                batch_size=50, 
                timesteps=None, 
                log_all_timesteps=False, 
                return_as_atoms_list=True,
                temp=temp,
                dataset_kwargs=dataset_kwargs
            )
            atoms_list = [sample[0] for sample in result_samples]
            write(filename=os.path.join(model, f"samples_{temp}.traj"), images=atoms_list)

if __name__ == "__main__":
    main()