from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.post_processing import filter_identical_structures, apply_rotation
from scipy.spatial.distance import pdist
from ase.io import read, write
import numpy as np
import random
import os



def main():
    random.seed(42)
    miller_index = "111"
    opt_method = "genetic_algorithm"
    db_files = [f"rnd_seed_{i+10}_samples.db" for i in range(50)]
    stored_datadicts = []
    template_atoms = None
    for db_file in db_files:
        db = Database.establish_connection(
            filename=db_file,
            pth_header=os.path.join("..", opt_method, "results_saas_fix_111")
        )
        template_atoms = db.template_atoms_surf
        stored_datadicts+=load_datadicts_from_db(database=db)
    
    filter_rotations = [False]
    for filter_rotation in filter_rotations:
        filtered_datadicts = filter_identical_structures(
            datadicts=stored_datadicts,
            filter_symmetry_equivalent=filter_rotation,
            miller_index=miller_index
        )
        db_save = Database.establish_connection(
            filename=f"{opt_method}_with_rot_filter.db" if filter_rotation else f"{opt_method}_no_rot_filter.db",
            database_kwargs={"append":False, "template_atoms_surf":template_atoms}
        )
        datadicts_for_save = []
        for datadict in filtered_datadicts:
            save_dict = {"elements":datadict["elements"]}
            score_dict = {}
            for key in ["rate", "e_form"]:
                if datadict[key] is not None:
                    score_dict[key] = datadict[key]
            save_dict["score_dict"] = score_dict    
            datadicts_for_save.append(save_dict)
        db_save.write_data_to_tables(data_dicts=datadicts_for_save)
        db_save.close_connection()


if __name__ == "__main__":
    main()