import os
import sys
sys.path.insert(0, "/home/kwalz/speciale_files/")

from gen_catalyst_toolkit.bash_scripter import ArrayJob


def main():
    model="WWL-GPR"
    batch_size = 50
    batches = [i+1 for i in range(15)]
    run_ID = 0 #equivalent to each runs random seed
    miller_index = "100"
    n_batches = 1
    opt_method = "bayesian_opt"
    
    job = ArrayJob(
        job_name="bayes_opt",
        n_jobs=len(batches),           
        partition="q48",
        mem_per_cpu="10G",
        cpus_per_task=1,
        walltime="12:00:00"
    )

    job.add_python_script(file_name="run_optimization.py", pth_header="..")
    job.add_param_file(filename="script_params.txt")
    job.write_bash_script(bash_file_name="bayesian_opt.sh")

    out_dir = f"results/miller_index_{miller_index}"
    if os.path.exists(out_dir):
        pass
    else:
        os.makedirs(out_dir)

    script_param_list = []
    for n_batches in batches:
        out_dir_new = os.path.join(out_dir, "n_"+str(n_batches))
        if not os.path.exists(out_dir_new):
            os.mkdir(out_dir_new)
        script_params = {
            "-rnd_seed":run_ID,
            "-m_index":miller_index,
            "-n_batches":n_batches,
            "-batch_size":batch_size,
            "-out":out_dir_new,
            "-opt":opt_method,
            "-model":model,
            "-time":"True"
            }
        script_param_list.append(script_params)
        
    write_param_file(filename="script_params.txt", script_params_list=script_param_list)


def convert_walltime_to_seconds(walltime):
    hours, minutes, seconds = walltime.split(":")
    return 60**2*int(hours) + 60*int(minutes) + int(seconds)

def write_param_file(filename:str, script_params_list:list):
    with open(file=filename, mode="w") as fileobj:
        for script_param_dict in script_params_list:
            new_line = ""
            for script_param in script_param_dict:
                new_line += f"{script_param}={script_param_dict[script_param]} "
            fileobj.write(f"{new_line}\n")

def get_sample_estimate():
    pass

if __name__ == "__main__":
    main()