from ase.io import read, write


def main():
    atoms_list = read("GeneticAlgorithm.traj", index=":")
    rate_min = 16.0
    stored_samples = []
    stored_keys = []
    for atoms in atoms_list:
        key = "".join(atoms.get_chemical_symbols())
        if key in stored_keys:
            pass
        else:
            stored_keys.append(key)
            stored_samples.append(atoms)
    high_rate_filtered = []
    for atoms in stored_samples:
        rate = atoms.info["rate"]
        if rate >= rate_min:
            high_rate_filtered.append(atoms)
    write("no_duplicates.traj", stored_samples)
    write("high_rates_no_duplicates.traj", high_rate_filtered)

if __name__ == "__main__":
    main()