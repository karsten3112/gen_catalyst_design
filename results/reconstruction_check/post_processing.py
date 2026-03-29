import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os
from ase_ml_models.utilities import (
    get_connectivity,
    get_edges_list_from_connectivity,
    get_connectivity_from_edges_list
)


def main():
    models = [
        "chgnet",
        "mace_mh1"
    ]

    element_pools = [
        "full",
        "all_fcc",
        "au_close_fcc",
        "ni_close_fcc"
    ]

    connectivity_kwargs = dict(
        method="ase",
        bond_cutoff=1.0,
        remove_pbc=True,
        skin=0.3
    )

    print("-------------RECONSTRUCTION CHECKS-------------")
    print(f"""using skin={connectivity_kwargs["skin"]}""")


    for model in models:
        print(f"---MODEL:{model}---")
        for element_pool in element_pools:
            filename = os.path.join(model, f"{element_pool}.traj")
            atoms_list = read(filename=filename, index=":")
            ordered_atoms_dict = order_init_final(atoms_list=atoms_list)
            reconstruction_checks = []
            for sample_num in ordered_atoms_dict:
                atoms_init, atoms_final = ordered_atoms_dict[sample_num][0], ordered_atoms_dict[sample_num][1]
                has_reconstructed = reconstruction_check(
                    atoms_init=atoms_init,
                    atoms_final=atoms_final,
                    connectivity_kwargs=connectivity_kwargs
                )
                reconstruction_checks.append(has_reconstructed)
            reconstruction_checks = np.array(reconstruction_checks, dtype=int)
            recon_count = np.sum(reconstruction_checks)
            recon_percent = recon_count/len(reconstruction_checks)
            print(f"For {element_pool} | amount of reconstructed surfaces: {recon_percent*100:3f}")
            #exit()

def order_init_final(
        atoms_list:list
    ):
    ordered_atoms_dict = {}
    for atoms in atoms_list:
        index = atoms.info["sample_num"]
        if index in ordered_atoms_dict:
            ordered_atoms_dict[index].append(atoms)
        else:
            ordered_atoms_dict[index] = [atoms]
    return ordered_atoms_dict

def reconstruction_check(
        atoms_init,
        atoms_final,
        connectivity_kwargs:dict={}
    ):
    init_connectivity = get_connectivity(
                atoms=atoms_init,
                **connectivity_kwargs
            )
    final_connectivity = get_connectivity(
                atoms=atoms_final,
                **connectivity_kwargs
            )
    con_diff = init_connectivity - final_connectivity
    mask = np.bool(np.abs(con_diff))
    if True in mask:
        return True
    else:
        return False



if __name__ == "__main__":
    main()