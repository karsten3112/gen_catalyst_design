from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.db import Database
from gen_catalyst_design.optimization import setup_optimization_objective, evaluate_score_from_symbols, Logger
from ase.io import read, write
import numpy as np
import random
import torch
import os


def main():
    random_seed = 42
    template_type = "cluster"
    miller_index = "100"
    diff_models = ["model_3", "model_8"]
    device = "cpu"
    use_log = True
    model_type = "best"
    model_pth_header = "../../../results/diffusion_model_late_approach/trained_models"
    train_atoms_list = read("../../../results/diffusion_model_parameters/datasets_100/genetic_algorithm_8000_no_saas.traj", ":")

    n_samples_per_cond = 2
    guidance_scale_dict = {"rate":2.0}

    rate_conditions = get_rate_conditions(
        atoms_list=train_atoms_list,
        use_log=use_log,
        percentiles=np.linspace(0.9, 0.99, 5)
    )
   
    reaction_mechanism, stabilizer, _ = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header="../../../databases",
        yaml_files_header="../../../yaml_files",
        include_stability=False,
    )

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    for diff_model in diff_models:
        if not os.path.exists(diff_model):
            os.makedirs(diff_model)

        ckpt_file = get_ckpt_file(
            diff_model_name=diff_model,
            model_type=model_type,
            pth_header=model_pth_header
        )
        diffusion_model = DiffusionModel.load_from_checkpoint(ckpt_file, weights_only=False).to(device=torch.device(device=device))
        for i, rate_condition in enumerate(rate_conditions):
            result_samples = diffusion_model.sample(
                n_samples=n_samples_per_cond,
                template_atoms=reaction_mechanism.clean_surface,
                conditioning_dicts=[{"rate":rate_condition} for _ in range(n_samples_per_cond)],
                guidance_scale_dict=guidance_scale_dict,
                condition_keys=["rate"],
                batch_size=20,
                timesteps=None, 
                log_all_timesteps=True, 
                return_as_atoms_list=True,
                temp=1.0,
                dataset_kwargs={"graph_kwargs":{"use_log":False}} 
            )
            for j, denoise_traj in enumerate(result_samples):
                write(
                    filename=os.path.join(diff_model, f"traj_{j}_condition_{i}.traj"),
                    images=denoise_traj
                )


            database = Database.establish_connection(
                filename=f"condition_{i}.db",
                pth_header=diff_model,
                database_kwargs={"append":False, "template_atoms_surf":reaction_mechanism.clean_surface}
            )

            logger = Logger(
                database=database,
                log_interval=10,
                match_log_interval_gen_iter=False
            )

            for sample_traj in result_samples:
                score = evaluate_score_from_symbols(
                    symbols=sample_traj[-1].get_chemical_symbols(),
                    reaction_mechanism=reaction_mechanism,
                    logger=logger,
                    stabilizer=stabilizer,
                    objective_key="rate"
                )

            logger.write_data_to_file()
            database.close_connection()

def get_ckpt_file(
        diff_model_name:str,
        model_type:str="best",
        pth_header:str=None
    ):
    model_type_kw_dict = {
        "best":"val_loss",
        "last":"last"
    }

    if pth_header is not None:
        file_dir = os.path.join(pth_header, diff_model_name)
    else:
        file_dir = diff_model_name
    
    ckpt_dir = os.path.join(file_dir, "checkpoints")
    ckpt_files = os.listdir(ckpt_dir)
    if model_type in model_type_kw_dict:
        kw = model_type_kw_dict[model_type]
        for ckpt_file in ckpt_files:
            if kw in ckpt_file:
                return os.path.join(ckpt_dir, ckpt_file)
    else:
        for ckpt_file in ckpt_files:
            if ckpt_file == model_type:
                return os.path.join(ckpt_dir, ckpt_file)
            

def get_rate_conditions(
        atoms_list:list,
        use_log:bool=True,
        percentiles:np.array=None,
        percentile_kwargs:dict={}
    ):
    rates = np.array([atoms.info["rate"] for atoms in atoms_list])
    if use_log:
        rates = np.log10(rates)

    if percentiles is None:
        percentiles = np.linspace(
            percentile_kwargs.pop("lower_percent_lim",0.8), 
            percentile_kwargs.pop("upper_percent_lim",0.95), 
            percentile_kwargs.pop("num_conds", 5)
        )
    conditions = np.array([np.round(np.percentile(rates, percentile*100), 2) for percentile in percentiles])
    return conditions


if __name__ == "__main__":
    main()