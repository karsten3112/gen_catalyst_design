

from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file

def main():
    num_seeds = 10
    n_jobs = num_seeds

    job = ArrayJob(
        job_name="rnd_search",
        n_jobs=n_jobs,
        walltime="10:00:00",
        partition="qany",
        mem_per_cpu="2G"
    )

    script_params_list = []
    for i in range(num_seeds):
        script_params = {
            "-filename":f"rnd_{i}_seed.db",
            "-setup_files_header":"../../..",
            "-rnd_seed":i
        }
        script_params_list.append(script_params)
    
    write_param_file(
        filename="script_params.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="run_random_search.py"
    )

    job.add_param_file(
        filename="script_params.txt"
    )

    job.write_bash_script(
        bash_file_name="rnd_search.sh"
    )    


if __name__ == "__main__":
    main()