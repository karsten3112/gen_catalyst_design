from ase.io import read
import os


def main():
    models = [
        "model_001",
        "model_002",
        #"model_003",
        "model_004",
        "model_005",
        #"model_006",
        #"model_007",
        #"model_008",
        #"model_009",
        #"model_015",
        #"model_017",
        #"model_018",
        #"model_019"
    ]
    
    active_sites = [0,1,2,3]

    for model in models:
        samples_file = os.path.join(model, "samples.traj")
        samples = read(samples_file, index=":")
        counter = 0
        for atoms in samples:
            symbols = atoms[active_sites].get_chemical_symbols()
            cu_num, pd_num = symbols.count("Cu"), symbols.count("Pd")
            if cu_num == 3 and pd_num == 1:
                counter+=1
        print(counter/100)
        #print(samples[0].info.keys())


if __name__ == "__main__":
    main()