#!/bin/bash
import inspect
import os
import sys


class Job:
    def __init__(self,
                 job_name:str,
                 partition="qgpu",
                 mem_per_cpu="6G",
                 cpus_per_task=1,
                 walltime="02:00:00",
                 error_out_file="job.err",
                 output_file="job.out",
                 ) -> None:
        
        default_script_params = {"job-name":job_name,
                                 "partition":partition,
                                 "mem-per-cpu":mem_per_cpu,
                                 "cpus-per-task":cpus_per_task,
                                 "time":walltime,
                                 "error":error_out_file,
                                 "output":output_file}

        self.script_params = {"default": default_script_params}

    def add_module(self, name, version):
        if self.script_params["modules"] is None:
            self.script_params["modules"] = {name: {"version":version}}
        else:
            self.script_params["modules"][name] = {"version":version}

    def add_source(self, name, dir, force=False):
        if self.script_params["sources"] is None:
            self.script_params["sources"] = {name: {"dir":dir, "force":force}}
        else:
            self.script_params["sources"][name] = {"dir":dir, "force":force}
    
    def write_sources(self, fileobj):
        parameter_dict = self.script_params["sources"]
        for param_type in parameter_dict:
            param_info = parameter_dict[param_type]
            if "dir" in param_info:
                line = f"{param_info['dir']}{param_type}"
            else:
                line = param_type
            if "force" in param_info:
                if param_info["force"] is True:
                    fileobj.write(f"source {line} --force\n")
                else:
                    fileobj.write(f"source {line}\n")
        fileobj.write("\n")
        return fileobj
        
    def write_modules(self, fileobj):
        parameter_dict = self.script_params["modules"]
        for param_type in parameter_dict:
            param_info = parameter_dict[param_type]
            if "version" in param_info:
                line = f"{param_type}/{param_info['version']}"
            else:
                line = param_type
            fileobj.write(f"module load {line}\n")
        fileobj.write("\n")
        return fileobj

    def add_python_script(self, file_name, pth_header=None, python_version="default", script_inputs:dict=None):
        if pth_header is not None:
            file_name = os.path.join(pth_header, file_name)

        if python_version == "default":
            self.script_params["python_script"] = {"file_name":file_name, "python_version":"python3"}
        else:
            self.script_params["python_script"] = {"file_name":file_name, "python_version":python_version}

        if script_inputs is not None:
            self.script_params["python_script"].update({"script_inputs":script_inputs})

    def write_python_script(self, fileobj):
        if "python_script" not in self.script_params:
            raise Exception("No python script has been assigned")
        
        script_settings = self.script_params["python_script"]
        inputs_line = ""
        if "python_version" in script_settings:
            inputs_line+=f"{script_settings['python_version']} "
        if "file_name" in script_settings:
            inputs_line+=f"{script_settings['file_name']} "
        
        if "script_inputs" in self.script_params:
            script_inputs = self.script_params["script_inputs"]
            for script_input in script_inputs:
                inputs_line+=f"{script_input}={script_inputs[script_input]} "
        fileobj.write(f"{inputs_line}")
        return fileobj

    def update_script_params(self, param_type, parameter_dict):
        if param_type not in self.script_params:
            self.script_params[param_type] = parameter_dict
        else:
            self.script_params[param_type].update(parameter_dict)

    def write_default_parameters(self, fileobj, outdir=None):
        parameter_dict = self.script_params["default"].copy()
        if outdir is not None:
            parameter_dict["output"] = f"{outdir}/{parameter_dict['output']}"
            parameter_dict["error"] = f"{outdir}/{parameter_dict['error']}"
        for param in parameter_dict:
            fileobj.write(f"#SBATCH --{param}={parameter_dict[param]}\n")
        return fileobj

    def write_bash_script(self, bash_file_name:str, outdir=None, return_file_obj=True):
        if outdir is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            bash_file_name = os.path.join(outdir, bash_file_name)
        fileobj = open(file=bash_file_name, mode="w")
        fileobj.write("#!/bin/bash\n")
        fileobj = self.write_default_parameters(fileobj=fileobj, outdir=outdir)
        fileobj.write("\n")

        if return_file_obj is True:
            return fileobj
        else:
            fileobj.close()
            return None


class ArrayJob(Job):
    def __init__(self, job_name, n_jobs:int, partition="qgpu", mem_per_cpu="6G", cpus_per_task=1, walltime="02:00:00", error_out_file="job.err", output_file="job.out"):
        super().__init__(job_name, partition, mem_per_cpu, cpus_per_task, walltime, error_out_file, output_file)
        self.script_params["default"]["array"] = f"1-{n_jobs}"
        self.param_file = None

    def add_param_file(self, filename:str, pth_header:str=None):
        if pth_header is not None:
            filename = os.path.join(pth_header, filename)
        self.param_file = filename
        
    def write_bash_script(self, bash_file_name, outdir=None, return_file_obj=False):
        fileobj = super().write_bash_script(bash_file_name, outdir, return_file_obj=True)
        if outdir is not None:
            fileobj.write(f'echo "========= Job started  at `date` ==========" >> {outdir}/{self.script_params["default"]["output"]}\n')
        else:
            fileobj.write(f'echo "========= Job started  at `date` ==========" >> {self.script_params["default"]["output"]}\n')
        fileobj.write("\n")

        fileobj.write('echo "My jobid: $SLURM_JOB_ID"\n')
        fileobj.write('echo "My array id: $SLURM_ARRAY_TASK_ID"\n')

        fileobj = self.write_python_script(fileobj=fileobj)
        if self.param_file is not None:
            fileobj.write(f'`awk "NR == $SLURM_ARRAY_TASK_ID" {self.param_file}`')
            fileobj.write("\n")
        else:
            fileobj.write("\n")

        fileobj.write("\n")
        if outdir is not None:
            fileobj.write(f'echo "========= Job Finished  at `date` ==========" >> {outdir}/{self.script_params["default"]["output"]}\n')
        else:
            fileobj.write(f'echo "========= Job Finished  at `date` ==========" >> {self.script_params["default"]["output"]}\n')

        if return_file_obj is True:
            return fileobj
        else:
            fileobj.close()
            return None


class DFT_Job(Job):
    def __init__(self, job_name: str,
                 mode="default",
                 run_scratch=False,
                 ) -> None:
        super().__init__(job_name)

        self.run_scratch = run_scratch

        if mode == "default":
            self.script_params["sources"] = {"modules.sh":{"dir":"/comm/swstack/bin/", "force":True},
                                             "bashrc":{"dir":"~/.", "force":False}
                                             }
            self.script_params["modules"] = {"intel":{"version":"2020.1"},
                                             "openmpi":{"version":"4.0.3"},
                                             "qe":{"version":"6.5"}
                                             }
        else:
            self.script_params["modules"] = None
            self.script_params["sources"] = None

    def write_bash_script(self, 
                          bash_file_name: str, 
                          dft_params_kw="DFT_settings", 
                          outdir=None):
        fileobj = super().write_bash_script(bash_file_name, outdir, return_file_obj=True)

        fileobj = self.write_sources(fileobj=fileobj)
        fileobj = self.write_modules(fileobj=fileobj)

        if self.run_scratch:
            pass #Needs to implement this
        fileobj.write("export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n")
        fileobj.write("\n")
        fileobj.write(f'echo "========= Job started  at `date` ==========" >> {outdir}/{self.script_params["default"]["output"]}\n')
        fileobj.write("(\n")
        
        fileobj = self.write_python_script(fileobj=fileobj)

        fileobj.write(")\n")
        fileobj.write(f'echo "========= Job Finished  at `date` ==========" >> {outdir}/{self.script_params["default"]["output"]}\n')
        if self.run_scratch:
            pass #Needs to implement this
        fileobj.close()

class OptimizerJob(Job):
    def __init__(self, job_name, walltime="02:00:00"):
        super().__init__(job_name=job_name, partition="qany", walltime=walltime)

    def write_bash_script(self, bash_file_name, outdir=None, return_file_obj=True):
        fileobj = super().write_bash_script(bash_file_name, outdir, return_file_obj=True)

        fileobj.write("\n")
        fileobj.write(f'echo "========= Job started  at `date` ==========" >> {outdir}/{self.script_params["default"]["output"]}\n')
        fileobj.write("(\n")

        fileobj = self.write_python_script(fileobj=fileobj)


def write_param_file(filename:str, script_params_list:list):
    with open(file=filename, mode="w") as fileobj:
        for script_param_dict in script_params_list:
            new_line = ""
            for script_param in script_param_dict:
                new_line += f"{script_param}={script_param_dict[script_param]} "
            fileobj.write(f"{new_line}\n")
