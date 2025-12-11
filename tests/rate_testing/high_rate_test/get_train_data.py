from ase_ml_models.databases import get_atoms_list_from_db
from gen_catalyst_design.db import Database, load_data_from_db
from ase.db import connect
from ase.io import write


def main():
    rate_min = 16.0
    opt_method = "GeneticAlgorithm"
    miller_index = "100"

    db = connect(f"../../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]
    print(template_atoms.info)

    exit()

    pth_header = f"../../../results/{opt_method}/results/Rh_Cu_Au_Pd/miller_index_{miller_index}"
    datadicts = []
    for runid in [0,1,2]:
        filename = f"runID_{runid}_results.db"
        database = Database.establish_connection(filename=filename, miller_index=miller_index, pth_header=pth_header)
        datadicts += load_data_from_db(database=database)
    
    filtered_dicts = filter_data_dicts(data_dicts=datadicts, rate_min=rate_min)
    #rates = [filtered_dict["rate"] for filtered_dict in filtered_dicts]
    atoms_list = []
    for filtered_dict in filtered_dicts:
        atoms = template_atoms.copy()
        atoms.symbols = filtered_dict["elements"]
        atoms.info["class"] = 0
        atoms.info["rate"] = filtered_dict["rate"]
        atoms_list.append(atoms)

    write("training_set.traj", atoms_list)
    


def filter_data_dicts(data_dicts:list, rate_min:float=None):
    if rate_min is None:
        return data_dicts
    else:
        filtered_datadicts = []
        for datadict in data_dicts:
            if datadict["rate"] < rate_min:
                pass
            else:
                filtered_datadicts.append(datadict)
        return filtered_datadicts

if __name__ == "__main__":
    main()