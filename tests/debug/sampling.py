from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
import os
import torch
from ase.io import write
from ase.db import connect
from ase_ml_models.databases import get_atoms_list_from_db


def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    miller_index = "100"
    models = [
        #"model_001",
        #"model_002"
        "model_002"
    ]

    masked_classes_model = "hej"

    guidance_scales = [0.6, 0.8, 2.0]#1.0, 1.2, 2.0]
    classes = [0, 1, 2]
    num_samples = 10

    template_db = connect(f"../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=template_db)[0]

    for model in models:
        out_dir = os.path.join("samples", model)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        checkpoint_file = os.path.join("AbsorbingStateModel", model, "checkpoints", "last.ckpt")
        diff_model = DiffusionModel.load_from_checkpoint(checkpoint_path=checkpoint_file)
        diff_model = diff_model.to(device=torch.device("cpu"))

        for cls in classes if model != masked_classes_model else [0]:
            conditionings = cls*torch.ones(size=(num_samples,), dtype=torch.long)
            for guidance_scale in guidance_scales:
                out_dir_final = os.path.join(out_dir, f"class_{cls}", f"g_{guidance_scale}_scale")
                if not os.path.exists(out_dir_final):
                    os.makedirs(out_dir_final)
                denoise_trajes = diff_model.sample(
                    n_samples=num_samples,
                    conditionings=conditionings,
                    template_atoms=template_atoms,
                    guidance_scale=guidance_scale,
                    return_as_atoms_list=True,
                    log_all_timesteps=True
                )
                for i, denoise_traj in enumerate(denoise_trajes):
                    write(os.path.join(out_dir_final,f"denoise_{i}.traj"), denoise_traj)



if __name__ == "__main__":
    main()