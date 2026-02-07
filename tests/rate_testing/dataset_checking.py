from ase.io import read, write
import numpy as np


def main():
    atoms_list = read("filtered_samples.traj", index=":")
    rates = np.array([atoms.info["rate"] for atoms in atoms_list])
    print(np.var(rates))
    indices = np.argwhere(rates > 24.0).squeeze()
    atoms_filtered = [atoms_list[index] for index in indices]
    print(rates[indices])
    write("saved.traj", atoms_filtered)
    min_index, max_index = np.argmin(rates), np.argmax(rates)
    print(min_index, max_index)
    write("min_max.traj", [atoms_list[min_index],atoms_list[max_index]])



if __name__ == "__main__":
    main()