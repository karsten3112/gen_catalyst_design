from ase.io import read, write
import numpy as np

def main():
    samples = read("high_rate_structs.traj", index=":")
    rates = np.array([sample.info["rate"] for sample in samples])
    classes = [sample.info["class"] for sample in samples]
    cls_dict = {}
    for cls, atoms in zip(classes, samples):
        if cls in cls_dict:
            cls_dict[cls].append(atoms)
        else:
            cls_dict[cls] = [atoms]
    for cls in cls_dict:
        rates = np.array([sample.info["rate"] for sample in cls_dict[cls]])
        print(np.mean(rates))
        write(filename=f"class_{cls}.traj", images=cls_dict[cls])
    
    
    #print(rates)
    #print(classes)
    #print(np.mean(rates))
    #filtered_samples = [atoms for atoms in samples if atoms[0].symbol == "Pd"]
    #write("filtered_samples.traj", filtered_samples)


if __name__ == "__main__":
    main()