from ase.io import read, write
import numpy as np
from ase_ml_models.utilities import (
    get_connectivity,
    get_edges_list_from_connectivity,
    get_connectivity_from_edges_list
)

def main():
    atoms_list = read("no_elem_swap.traj", index=":")
    init_atoms, relaxed_atoms = atoms_list[0],atoms_list[-1]
    connectivity_kwargs = dict(
        method="ase",
        bond_cutoff=1.0,
        remove_pbc=True,
        skin=0.2
    )
    init_con = get_connectivity(
        atoms=init_atoms,
        **connectivity_kwargs
    )

    relaxed_con = get_connectivity(
        atoms=relaxed_atoms,
        **connectivity_kwargs
    )
    print(init_con[9])
    print(relaxed_con[9])

    diff = relaxed_con - init_con
    mask = np.bool(np.abs(diff))
    recon_indices = np.argwhere(mask == True)
    #print(recon_indices)
    #print(diff[recon_indices])
    #init_positions = init_atoms.positions
    #relaxed_positions = relaxed_atoms.positions
    #diff = np.linalg.norm(relaxed_positions-init_positions, axis=1)
    #print(diff)

if __name__ == "__main__":
    main()