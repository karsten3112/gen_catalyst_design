from gen_catalyst_design.db import Database, load_data_from_db
from ase.io import write
from ase.db import connect
from ase_ml_models.databases import get_atoms_list_from_db
import numpy as np



def main():
    miller_index = "100"
    opt_method = "random_search"
    pth_header = f"../../results/{opt_method}/results/Rh_Cu_Au_Pd/miller_index_{miller_index}"
    assign_no_class = False
    rate_min = 1.0

    num_classes = {}
    datadicts = []
    for runid in [0,1,2]:
        filename = f"runID_{runid}_results.db"
        database = Database.establish_connection(filename=filename, miller_index=miller_index, pth_header=pth_header)
        datadicts += load_data_from_db(database=database)
    filtered_dicts = filter_data_dicts(data_dicts=datadicts, rate_min=rate_min)
    rates = [data["rate"] for data in filtered_dicts]
    rate_max = np.max(rates)
    step = 2.5
    class_ranges = np.arange(rate_min, np.ceil(rate_max)+step, step)
    num_classes[opt_method] = len(class_ranges) - 1
    assign_rate_class(
        data_dicts=filtered_dicts, 
        class_ranges=class_ranges, 
        assign_no_class=assign_no_class
    )

    atoms_list = []
    db = connect("../../databases/cluster_templates/100_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]

    for datadict in filtered_dicts:
        print(datadict.keys())
        atoms = template_atoms.copy()
        atoms.info["rate"] = datadict["rate"]
        atoms.info["class"] = datadict["class"]
        atoms.symbols = datadict["elements"]
        atoms_list.append(atoms)
    print(f"num of classes generated: {num_classes}")
    write(f"{opt_method}.traj", images=atoms_list)





def assign_rate_class(data_dicts:list, class_ranges:np.array, assign_no_class:bool=False):
    for datadict in data_dicts:
        rate = datadict["rate"]
        indices = np.argwhere(rate > class_ranges)
        if assign_no_class:
            datadict.update({"class":0})
        else:
            datadict.update({"class":len(indices)-1})

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