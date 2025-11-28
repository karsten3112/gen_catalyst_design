import os
import sys
sys.path.insert(0, "/home/kwalz/speciale_files/")

from gen_catalyst_toolkit.bash_scripter import ArrayJob


def main():
    model="WWL-GPR"
    run_IDs = [i for i in range(100)] #equivalent to each runs random seed
    element_pools = ["Rh,Cu,Pd", "Rh,Pd,Au", "Pd,Cu,Au", "Rh,Cu,Au,Pd"]
    miller_index = "100"
    n_batches = 100
    batch_size = 100
    opt_method = "random_search"
    
    job = ArrayJob(
        job_name="rnd_search",
        n_jobs=4*len(run_IDs),           
        partition="q48",
        mem_per_cpu="1G",
        cpus_per_task=1,
        walltime="06:00:00"
    )

    job.add_python_script(file_name="run_optimization.py", pth_header="..")
    job.add_param_file(filename="script_params.txt")
    job.write_bash_script(bash_file_name="random_search.sh")

    script_param_list = []
    for element_pool in element_pools:
        elem_string = "_".join(element_pool.split(","))
        out_dir = f"results/{elem_string}/miller_index_{miller_index}"
        if os.path.exists(out_dir):
            pass
        else:
            os.makedirs(out_dir)

        for run_ID in run_IDs:
            script_params = {
                "-rnd_seed":run_ID,
                "-m_index":miller_index,
                "-n_batches":n_batches,
                "-batch_size":batch_size,
                "-out":out_dir,
                "-opt":opt_method,
                "-model":model,
                "-elem_pool":element_pool,
                "-time":"False"
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