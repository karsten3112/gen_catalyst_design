from ase.atoms import Atoms
from ase_ml_models.databases import get_atoms_list_from_db
from ase.db import connect
from ase.io import write
import random
import numpy as np

def main():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    miller_index = "100"

    template_db = connect(f"../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=template_db)[0]
    class_pool = ["Pd", "Au", "Ni"]
    class_prob = 0.8
    samples_per_class = 200
    atoms_list = []
    for i, _ in enumerate(class_pool):
        probs = np.ones(len(class_pool))*(1.0-class_prob)/(len(class_pool)-1)
        probs[i] = class_prob
        for _ in range(samples_per_class):
            atoms = template_atoms.copy()
            symbols = random.choices(population=class_pool, weights=probs, k=len(atoms))
            atoms.symbols = symbols
            atoms.info["class"] = i
            atoms_list.append(atoms)
    
    write("dataset.traj", atoms_list)    



    


if __name__ == "__main__":
    main()