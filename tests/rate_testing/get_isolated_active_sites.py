from ase.io import read, write
from ase.atoms import Atoms

def main():
    samples = read("high_rate_structs.traj", index=":")
    active_sites = [0,1,2,3,4]
    isolated_atoms_list = []
    for atoms in samples:
        isolated_atoms = Atoms(atoms[active_sites])
        isolated_atoms.info = {}
        rows = atoms.info["connectivity"][active_sites]
        isolated_connectivity = rows[:,active_sites]
        isolated_atoms.info["connectivity"] = isolated_connectivity
        isolated_atoms_list.append(isolated_atoms)

    write("isolated_active_sites.traj", images=isolated_atoms_list)


if __name__ == "__main__":
    main()