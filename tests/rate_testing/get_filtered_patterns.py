from ase.io import read, write


def main():
    samples = read("high_rate_structs.traj", index=":")
    print(samples[0].info)
    #filtered_samples = [atoms for atoms in samples if atoms[0].symbol == "Pd"]
    #write("filtered_samples.traj", filtered_samples)


if __name__ == "__main__":
    main()