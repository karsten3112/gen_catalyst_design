
from gen_catalyst_design.bash_scripter import ArrayJob, write_param_file
import os




def main():
    reserve_gpu = True
    model_names = ["model_3"]#f"model_set_{i+1}" for i in range(3)]
    
    script_params = {
        "-dev": "cuda" if reserve_gpu else "cpu"
    }
    script_params_list = []
    i = 0
    for model_name in model_names:
        script_dict = script_params.copy()
        script_dict.update({
        "-m_name":model_name
        })
        script_params_list.append(script_dict)
        i+=1
    

    job = ArrayJob(
        job_name="diff_sample",
        n_jobs=i,
        walltime="12:00:00",
        reserve_gpu=reserve_gpu
    )

    write_param_file(
        filename="script_params_sample.txt",
        script_params_list=script_params_list
    )

    job.add_python_script(
        file_name="benchmark_sampling.py"
    )

    job.add_param_file(
        filename="script_params_sample.txt"
    )

    job.write_bash_script(
        bash_file_name="sampling.sh"
    )    


if __name__ == "__main__":
    main()