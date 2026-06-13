from ase.db import connect
from gen_catalyst_design.utils import get_full_element_pool_no_saas, get_periodic_surface
from ase_ml_models.databases import get_atoms_list_from_db
from gen_catalyst_design.stability import apply_inversion_symmetry
from ase.io import read, write
import random


def main():
    random.seed(42)
    template_types = ["surface"]
    facets = ["111"]
    element_pool = get_full_element_pool_no_saas()


    for template_type in template_types:
        for facet in facets:
            atoms, _ = get_periodic_surface(
                miller_index=facet,
                n_layers_z=4
                #a_lat=1.0
            )
            write("test.traj", [atoms])
            #db = connect(f"{template_type}_templates/{facet}_templates.db")
            #atoms = get_atoms_list_from_db(db)[0]
            #rnd_elements = random.choices(element_pool, k=len(atoms))
            #atoms.symbols = rnd_elements
            atoms_inv = apply_inversion_symmetry(
                atoms=atoms.copy(),
                miller_index=facet,
                a_lat=None
            )
            write("inv_test.traj", [atoms_inv])
            write(f"{facet}_inv_{template_type}_img_inv.png", images=[atoms_inv], **dict(rotation='10z,-75x'))


if __name__ == "__main__":
    main()