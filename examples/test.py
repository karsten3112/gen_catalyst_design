from ase.atoms import Atoms
from ase.io import write
from gen_catalyst_design.utils import get_periodic_surface
import random
import numpy as np

def main():
    elements = ["Rh", "Cu", "Au"] # Elements of the surface.
    random_seed = 42 # Random seed for reproducibility.
    np.random.seed(random_seed)
    random.seed(random_seed)
    surface, _ = get_periodic_surface(miller_index="100", vacuum=0.0, n_layers_z=4)
    symbols = surface.get_chemical_symbols()# random.choices(population=elements, k=len(surface))
    surface.symbols = symbols
    tot_symbols = []
    for _ in range(2):
        symbols.reverse()
        tot_symbols+=symbols

    write("surface_test.traj", surface)
    surface_mirror, _ = get_periodic_surface(miller_index="100", vacuum=15.0, n_layers_z=8)
    surface_mirror.symbols = tot_symbols
    write("surface_mirror.traj", surface_mirror)
    #surface = fcc100(symbol="Au", size=(3,3,8), vacuum=15)
   #bulk_atoms = bulk(name="Au", orthorhombic=True)
    #write("bulk_test.traj", bulk_atoms)


if __name__ == "__main__":
    main()