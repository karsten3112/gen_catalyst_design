from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file


def main():
    reserve_gpu = True
    noisers = [
        "AbsorbingStateNoiser",
        "UniformTransitionsNoiser"
    ]

    schedulers = [
        "LinearBetaScheduler",
        "ExponentialBetaScheduler",
        "CosineScheduler",
        "LinearAlphaScheduler"
    ]

    p_drops = [0.1, 0.2, 0.5]

    n_jobs = len(noisers)*len(schedulers)*len(p_drops)


    job = ArrayJob(
        job_name="hype_opt",
        n_jobs=n_jobs,
        walltime="04:00:00",
        reserve_gpu=reserve_gpu
    )

    script_params_list = []
    i = 1
    for noiser in noisers:
        for scheduler in schedulers:
            for p_drop in p_drops:
                script_params = {
                    "-m_name":f"model_{i:03d}",
                    "-noiser":noiser,
                    "-sched":scheduler,
                    "-p_drop":p_drop,
                    "-dev": "gpu" if reserve_gpu else "-cpu"
                }
                script_params_list.append(script_params)
                i+=1
    
    write_param_file(
        filename="script_params.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="hyperparams_search.py"
    )

    job.add_param_file(
        filename="script_params.txt"
    )

    job.write_bash_script(
        bash_file_name="hype_opt.sh"
    )    


if __name__ == "__main__":
    main()