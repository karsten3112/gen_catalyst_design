

from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file


def main():
    crossover_types = [
        #"single_point", 
        #"two_points",
        "uniform"
    ]
    hej = 2
    mutation_types = [
        #"random" 
        #"swap",
        "inversion",
        "scramble"
    ]

    n_jobs = len(crossover_types)*len(mutation_types)


    job = ArrayJob(
        job_name="gen_alg",
        n_jobs=n_jobs,
        walltime="06:00:00",
        partition="qany"
    )

    script_params_list = []
    for crossover_type in crossover_types:
        for mutation_type in mutation_types:
            script_params = {
                "-cross_type":crossover_type,
                "-mut_type":mutation_type,
                "-filename":f"{crossover_type}_{mutation_type}.db",
                "-setup_file_header":"../../../.."
            }
            script_params_list.append(script_params)
    
    write_param_file(
        filename="script_params.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="run_genetic_algorithm.py",
        pth_header="../"
    )

    job.add_param_file(
        filename="script_params.txt"
    )

    job.write_bash_script(
        bash_file_name="gen_alg.sh"
    )    


if __name__ == "__main__":
    main()