
from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file



def main():
    reserve_gpu = True
    noisers = [
        "AbsorbingStateNoiser",
        #"UniformTransitionsNoiser"
    ]

    schedulers = [
        #"LinearBetaScheduler",
        #"ExponentialBetaScheduler",
        "CosineScheduler",
        #"LinearAlphaScheduler"
    ]
    
    drop_probs = [0.1]#, 0.2, 0.5]

    script_params = {
        "-data_traj":"../datasets/genetic_algorithm_2000.traj",
        "-m_epochs":1000,
        "-patience":1000,
        "-out":"hyperparam_search",
        "-beta_max":1.0,
        "-beta_min":1e-4,
        "-dev":"gpu" if reserve_gpu else "cpu"
    }
    script_params_list = []
    i = 0
    for noiser in noisers:
        for scheduler in schedulers:
            for drop_prob in drop_probs:
                script_dict = script_params.copy()
                script_dict.update({
                    "-m_name":f"model_{i+1}",
                    "-noiser":noiser,
                    "-scheduler":scheduler,
                    "-p_drop":drop_prob
                })
                script_params_list.append(script_dict)
                if scheduler == "ExponentialBetaScheduler":
                    extra_beta_dict = script_dict.copy()
                    extra_beta_dict["-beta_max"] = 5e-2
                    extra_beta_dict["-beta_min"] = 1e-4
                    script_params_list.append(extra_beta_dict)
                    i+=1
                i+=1
    
    #n_jobs = len(noisers)*len(schedulers)*len(drop_probs)

    job = ArrayJob(
        job_name="diff_opt",
        n_jobs=i,
        walltime="12:00:00",
        reserve_gpu=reserve_gpu
    )

    write_param_file(
        filename="script_params.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="../train_diff_model.py"
    )

    job.add_param_file(
        filename="script_params.txt"
    )

    job.write_bash_script(
        bash_file_name="hype_opt.sh"
    )    


if __name__ == "__main__":
    main()