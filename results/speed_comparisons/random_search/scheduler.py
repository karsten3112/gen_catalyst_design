import os
import sys
sys.path.insert(0, "../../../")

from gen_catalyst_toolkit.utilities.bash_scripter import Job


def main():
    script_params = {"ntasks-per-node":[1,2,3,4,5,6],
                     "mem-per-cpu":["1G", "2G", "3G", "4G", "5G"],
                     "nodes":[1,2,3,4,5,6],
                     "partition":["q20","q24", "q28", "q36", "q40", "q48"]
                     }
    
    default_script_params = {"job_name":"rnd_search",
                            "partition":"qgpu",
                            "nodes":1,
                            "mem_per_cpu":"1G",
                            "cpus_per_task":1,
                            "walltime":"05:00:00",
                            "ntasks_per_node":1,
                            "error_out_file":"job.err",
                            "output_file":"job.out"}


    model="WWL-GPR"
    miller_index = "100"
    n_batches = 7
    batch_size = 100
    opt_method = "random_search"
    random_seed = 42

    out_dir = "results"
    if os.path.exists(out_dir):
            pass
    else:
        os.mkdir("results")
    
    for param in script_params:
        values = script_params[param]
        out_dir = f"results/timings_{param}"
        if os.path.exists(out_dir):
            pass
        else:
            os.mkdir(out_dir)

        for value in values:
            out_dir_new = os.path.join(out_dir, str(value))
            if os.path.exists(out_dir_new):
                 pass
            else:
                os.mkdir(out_dir_new)
            
            job = Job(**default_script_params)
            job.script_params["default"][param] = value
            job.add_python_script(file_name="run_optimization.py", pth_header="../../")
            python_script_params = {"-rnd_seed":random_seed,
                             "-m_index":miller_index,
                             "-n_batches":n_batches,
                             "-batch_size":batch_size,
                             "-out":out_dir_new,
                             "-opt":opt_method,
                             "-model":model
                             }
            job.update_script_params("script_inputs", python_script_params)
            fileobj = job.write_bash_script(bash_file_name="random_search.sh", return_file_obj=True)
            fileobj = job.write_python_script(fileobj=fileobj)
            fileobj.close()
            os.system(f"sbatch random_search.sh")

def convert_walltime_to_seconds(walltime):
    hours, minutes, seconds = walltime.split(":")
    return 60**2*int(hours) + 60*int(minutes) + int(seconds)

if __name__ == "__main__":
    main()