from gen_catalyst_design.optimization import setup_optimization_objective, evaluate_score_from_symbols, Logger
from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.db import Database
from ase.db import connect
from ase.io import read, write
import torch
import os
import random


def main():
    random_seed = 42
    n_samples = 100
    miller_index = "100"
    template_type = "cluster"
    ckpt_file_header = "full_surface/rnd_search_test_filtered"
    ckpt_file = os.path.join(ckpt_file_header,"checkpoints/last.ckpt") #epoch=epoch=146-val=val_loss=1.9309.ckpt
    include_stability = False
    temps = [1.0]
    rate_conditions = [10**(-2.5), 10**(-1.5), 10**(0.0), 10**(0.5)]#[3.0e-7, 4.5e-5, 1.0]#[5.0, 10.0, 15.0, 20.0]
    e_form_conditions = [4.0] * len(rate_conditions)
    guidance_scale = 2.0
    guidance_scale_dict = {
        #"joint":0.0,#guidance_scale,#guidance_scale,
        "rate":guidance_scale,
        #"e_form":0.0*guidance_scale
    }
    #rate_guidance = 0.0
    #e_form_guidance = 2.0

    reaction_mechanism, stabilizer, template_atoms_list = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header="../../databases",
        yaml_files_header="../../yaml_files",
        include_stability=include_stability,
    )

    diff_model = DiffusionModel.load_from_checkpoint(ckpt_file)
    diff_model = diff_model.to(device=torch.device("cpu"))
    i = 0
    for temp in temps:
        for rate_condition, e_form_condition in zip(rate_conditions, e_form_conditions):
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            print("===========================================================================================================")
            print(f"""Began sampling {n_samples} samples with condition: rate={rate_condition} & e_form={e_form_condition}""")
            #print(
            #    f"""Began sampling {n_samples} samples with condition: rate={rate_condition} & e_form={None} 
            #      | guidance scales: g_rate={rate_guidance} & g_e_form={None} 
            #      | temperature: temp={temp}"""
            #)
            result_samples = diff_model.sample(
                n_samples=n_samples,
                template_atoms=reaction_mechanism.clean_surface,
                conditioning_dicts=[{"rate":rate_condition} for _ in range(n_samples)],
                guidance_scale_dict=guidance_scale_dict,
                condition_keys=["rate"],
                batch_size=50,
                timesteps=None, 
                log_all_timesteps=False, 
                return_as_atoms_list=True,
                temp=temp,
                dataset_kwargs={"graph_kwargs":{"use_log":True}} 
            )

            save_dir = os.path.join(ckpt_file_header, "samples")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            database = Database.establish_connection(
                filename=f"joint_evals_{i+1}.db",
                pth_header=save_dir,
                database_kwargs={"append":False, "template_atoms_surf":reaction_mechanism.clean_surface}
            )

            logger = Logger(
                database=database,
                log_interval=10,
                match_log_interval_gen_iter=False
            )
            print("Began evaluation of rate from sampled structures")
            for sample in result_samples:
                score = evaluate_score_from_symbols(
                    symbols=sample[0].get_chemical_symbols(),
                    reaction_mechanism=reaction_mechanism,
                    logger=logger,
                    stabilizer=stabilizer,
                    objective_key="rate"
                )
            print("finished")
            i+=1




if __name__ == "__main__":
    main()