from gen_catalyst_design.db import Database, load_datadicts_from_db, load_atoms_list_from_db
from ase.io import write

def main():
    db = Database.establish_connection(
        "test_opt.db"
    )
    datadicts = load_datadicts_from_db(db)
    template_atoms = db.template_atoms_surf
    gen_iters = list(range(51))
    #print(gen_iters)
    #exit()
    atoms_list = []
    for datadict in datadicts:
        atoms = template_atoms.copy()
        gen_iter = datadict["gen_iter"]
        if gen_iter in gen_iters:
            atoms.info["rate"] = datadict["rate"]
            atoms.symbols = datadict["elements"]
            atoms_list.append(atoms)
    write("filtered_samples.traj", images=atoms_list)
    #unique_structs = []
    #for datadict in datadicts:
    #    elements = "".join(datadict["elements"])
    #    if elements not in unique_structs:
    #        unique_structs.append(elements)
    #print(len(unique_structs)/len(datadicts))


if __name__ == "__main__":
    main()