

from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file


def main():
    models = [
        "chgnet",
        "mace_mh1"
    ]

    element_pools = [
        "full",
        "all_fcc",
        "au_close_fcc",
        "ni_close_fcc"
    ]

    n_jobs = len(models)*len(element_pools)


    job = ArrayJob(
        job_name="relax",
        n_jobs=n_jobs,
        walltime="10:00:00",
        partition="qany"
    )

    script_params_list = []
    for model in models:
        for element_pool in element_pools:
            script_params = {
                "-elem_pool":element_pool,
                "-calc":model,
                "-file":f"{element_pool}.traj",
                "-out":model
            }
            script_params_list.append(script_params)
    
    write_param_file(
        filename="script_params.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="relax.py",
        pth_header="../"
    )

    job.add_param_file(
        filename="script_params.txt"
    )

    job.write_bash_script(
        bash_file_name="run_relax.sh"
    )    


if __name__ == "__main__":
    main()