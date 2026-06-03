from ase.io import read, write




def main():
    elements = ["Co", "Os", "Fe"]
    facets = ["100", "111"]
    for facet in facets:
        for element in elements:
            relax_traj = read(filename=f"{facet}_{element}.traj", index=":")
            for idx, name in zip([0,-1], ["init", "final"]):
                atoms = relax_traj[idx]
                write(f"{facet}_{element}_{name}.png", images=[atoms], **dict(rotation='10z,-75x'))



if __name__ == "__main__":
    main()