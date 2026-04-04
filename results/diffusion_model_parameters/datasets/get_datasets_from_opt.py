from gen_catalyst_design.db import Database, load_atoms_list_from_db
from ase.io import read, write
import os

def main():
    rnd_seed = 10
    num_sample_splits = [
        2000,
        5000
    ]
    opt_methods = [
        "random_search",
        "genetic_algorithm"
    ]

    for opt_method in opt_methods:
        db = Database.establish_connection(
            filename=f"rnd_seed_{rnd_seed}_samples.db",
            pth_header=os.path.join("../../optimization", opt_method, "results")
        )
        atoms_list_tot = load_atoms_list_from_db(
            database=db
        )
        for num_sample_split in num_sample_splits:
            write(f"{opt_method}_{num_sample_split}.traj", images=atoms_list_tot[0:num_sample_split])

if __name__ == "__main__":
    main()