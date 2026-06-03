
from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file



def main():
    reserve_gpu = True
    param_sets = {
        "with_att_pool_no_dist_no_log":{
            "noiser":"AbsorbingStateNoiser",
            "scheduler": "CosineScheduler"
        },
        #"8":{
        #    "noiser":"UniformTransitionsNoiser",
        #    "scheduler":"CosineScheduler"
        #}
    }
    

    script_params = {
        "-data_traj":"no_duplicates.traj",
        "-m_epochs":2000,
        "-out":"trained_models",
        "-beta_max":1.0,
        "-beta_min":1e-4,
        "-dev": "cuda" if reserve_gpu else "cpu",
        "-p_drop": 0.1,
        "-pat":100,
        "-lr":5e-4,
        "-log":False,
        "-log_proj_name":"att_pooling_importance"
    }
    script_params_list = []
    i = 0
    for param_set in param_sets:
        script_dict = script_params.copy()
        script_dict.update({
            "-m_name":f"model_{param_set}", #CHANGE THIS
            "-noiser":param_sets[param_set]["noiser"],
            "-sched":param_sets[param_set]["scheduler"]
        })
        script_params_list.append(script_dict)
        i+=1
    
    #n_jobs = len(noisers)*len(schedulers)*len(drop_probs)

    job = ArrayJob(
        job_name="diff_train",
        n_jobs=i,
        walltime="12:00:00",
        reserve_gpu=reserve_gpu
    )

    write_param_file(
        filename="script_params.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="train_diff_model.py",
    )

    job.add_param_file(
        filename="script_params.txt"
    )

    job.write_bash_script(
        bash_file_name="diff_train.sh"
    )    


if __name__ == "__main__":
    main()