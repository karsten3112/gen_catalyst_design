from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file
from ase_ml_models.yaml import write_to_yaml
import numpy as np
import os

def main():
    add_default_params = True
    num_samples = 2000
    random_seeds = [str(i) for i in range(10)]
    crossover_types = [
        "single_point", 
        "two_points",
        "uniform"
    ]
    mutation_types = [
        "random", 
        "swap",
        "inversion",
        "scramble"
    ]

    n_jobs = len(crossover_types)*len(mutation_types)


    if add_default_params:
        n_jobs+=1

    job = ArrayJob(
        job_name="gen_alg",
        n_jobs=n_jobs,
        walltime="24:00:00",
        partition="qany",
        mem_per_cpu="4G"
    )
    i = 0
    script_params_list = []
    hyperparam_set_dict = {}
    for cross_type in crossover_types:
        for mut_type in mutation_types:
            outdir = f"set_{i}"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            script_params = {
                "-cross_type":cross_type,
                "-mut_type":mut_type,
                "-selection_type":"sss", 
                "-rnd_seeds":",".join(random_seeds),
                "-dir":outdir,
                "-setup_files_header":"../../../gen_catalyst_design",
                "-n_samples":num_samples
            }

            hyperparam_set_dict[outdir] = script_params

            script_params_list.append(script_params)
            i+=1

    if add_default_params:
        if not os.path.exists("default"):
            os.makedirs("default")
        default_params = {
            "-mut_type":"random",
            "-cross_type":"single_point",
            "-selection_type":"sss",
            "-rnd_seeds":",".join(random_seeds), 
            "-dir":"default", 
            "-setup_files_header":"../../../gen_catalyst_design",
            "-n_samples":num_samples
        }
        script_params_list.append(default_params)
        hyperparam_set_dict["default"] = default_params

    write_to_yaml("hyperparams.yaml", hyperparam_set_dict)

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