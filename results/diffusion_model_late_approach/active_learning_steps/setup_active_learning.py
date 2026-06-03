from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.bash_scripter import Job
import os





def main():
    model_dir_header = "../../diffusion_model_parameters/8000k_sample_models_100_no_saas/trained_models"
    init_train_dataset = "genetic_alg_dataset_no_saas"
    model_name = "model_3"

    outdir = os.path.join(init_train_dataset, model_name+"_active_finale")

    #training_params
    g_scale_rate = 2.0
    reserve_gpu = True
    

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    get_pre_estim_sample_db(
        condition_dbs=[f"condition_{i}.db" for i in range(5)],
        pth_header=os.path.join(model_dir_header, model_name, "samples_42_seed"),
        outdir=outdir
    )

    ckpt_file, ckpt_pth_header = get_init_model_ckpt(
        model_name=model_name,
        pth_header=os.path.join(model_dir_header),
        model_type="best"
    )

    job = Job(
        job_name="act_learn",
        partition="qgpu" if reserve_gpu else "qany",
        reserve_gpu=reserve_gpu,
        walltime="12:00:00",
        error_out_file=os.path.join(outdir, "job.err"),
        output_file=os.path.join(outdir, "job.out")
    )

    script_inputs = {
        "-model_ckpt": os.path.join(ckpt_pth_header, ckpt_file),
        "-init_traj": os.path.join("../../diffusion_model_parameters/datasets_100", "genetic_algorithm_8000_no_saas.traj"),
        "-pre_sample_db": os.path.join(outdir, "pre_estim.db"),
        "-n_loops": 1,
        "-n_samples_per_loop": 1000,
        "-m_index": "100",
        "-proj_name": "active_learning_no_saas",
        "-out": outdir,
        "-dev": "cuda" if reserve_gpu else "cpu"
    }

    job.add_python_script(
        file_name="active_learning.py",
        pth_header="..",
        script_inputs=script_inputs
    )

    fileobj = job.write_bash_script(
        bash_file_name="run_active_learn.sh",
        return_file_obj=True
    )

    fileobj = job.write_python_script(fileobj=fileobj)

    fileobj.close()
    
    



def get_init_model_ckpt(
        model_name:str,
        pth_header:str=None,
        model_type:str="best"
    ):
    ckpt_pth = os.path.join(model_name, "checkpoints")
    if pth_header is not None:
        ckpt_pth = os.path.join(pth_header,ckpt_pth)
    ckpt_file_list = os.listdir(ckpt_pth)
    for ckpt_file in ckpt_file_list:
        if model_type in ckpt_file:
            return ckpt_file, ckpt_pth


def get_pre_estim_sample_db(
        condition_dbs:list,
        pth_header:str=None,
        outdir:str=None,
    ):
    tot_datadicts = []
    for condition_db in condition_dbs:
        database = Database.establish_connection(
            filename=condition_db,
            pth_header=pth_header
        )
        template_atoms = database.template_atoms_surf
        tot_datadicts+=load_datadicts_from_db(database=database)
    pre_estim_db = Database.establish_connection(
        filename="pre_estim.db",
        pth_header=outdir,
        database_kwargs={"template_atoms_surf":template_atoms, "append":False}

    )
    formatted_dicts = update_scoredict_format(datadicts=tot_datadicts)
    pre_estim_db.write_data_to_tables(data_dicts=formatted_dicts)
    pre_estim_db.close_connection()



def update_scoredict_format(
        datadicts:list
    ):
    formatted_dicts = []
    score_keys = ["rate", "e_form"]
    for datadict in datadicts:
        result_dict = {"score_dict":{}}
        for key in datadict:
            if key in score_keys:
                if datadict[key] is not None:
                    result_dict["score_dict"][key] = datadict[key]
            elif key in ["elements"]:
                result_dict[key] = datadict[key]
        formatted_dicts.append(result_dict)
    return formatted_dicts


if __name__ == "__main__":
    main()