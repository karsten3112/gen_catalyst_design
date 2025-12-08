from gen_catalyst_design.db import Database, load_data_from_db
from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.reaction_rates import ReactionMechanism
from ase.db import connect
from ase.io import write
from ase_ml_models.databases import get_atoms_list_from_db

import torch
import os

def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    db = connect("../../databases/templates/100/100_templates.db")
    template_atoms_list = get_atoms_list_from_db(db_ase=db)
    do_sampling = True

    opt_methods = [
        #"random_search",
        "GeneticAlgorithm"
    ]

    conditions = {
        "random_search":[0, 2, 4, 6],
        "GeneticAlgorithm":[7]#[1, 4, 7, 10]
    }


    guidance_scales = [0.8] #0.6, 1.0, 1.2, 2.0]
    n_samples = 200

    models = [
        "model_003"
    ]

    for opt_method in opt_methods:
        result_dir = f"samples/{opt_method}"
        model_str = opt_method + "_training_absorb"
        #models = os.listdir(path=model_str)
        for model in models:
            model_pth = os.path.join(model_str, model, "checkpoints")
            model_files = os.listdir(model_pth)
            model_filename = ""
            for model_file in model_files:
                if "epoch" in model_file:
                    model_filename = model_file
                    break
            diff_model = DiffusionModel.load_from_checkpoint(os.path.join(model_pth, model_filename))
            diff_model = diff_model.to(device=torch.device("cpu"))#torch.zeros(size=(10,), dtype=torch.long)#
            conditionings = conditions[opt_method]
            for condition in conditionings:
                for guidance_scale in guidance_scales:
                    final_dir = os.path.join(result_dir, model, f"class_{condition}")
                    if not os.path.exists(final_dir):
                        os.makedirs(final_dir)
                    if do_sampling:    
                        result_samples = diff_model.sample(
                            guidance_scale=guidance_scale,
                            n_samples=n_samples, 
                            conditionings=condition*torch.ones(size=(n_samples,), dtype=torch.long), 
                            template_atoms=template_atoms_list[0], 
                            batch_size=50, 
                            timesteps=None, 
                            log_all_timesteps=False, 
                            return_as_atoms_list=True
                        )
                        atoms_list = [sample[0] for sample in result_samples]
                        out_file = os.path.join(final_dir, f"g_{guidance_scale}_scale.traj")
                        write(filename=out_file, images=atoms_list)


if __name__ == "__main__":
    main()