from ase.io import read, write
from gen_catalyst_design.utils import get_atoms_from_template_db
from gen_catalyst_design.stability import apply_inversion_symmetry

def main():
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{100}_templates.db",
        pth_header="../databases/surface_templates"
    )
    clean_template = template_atoms_list[0]
    inverted_slab = apply_inversion_symmetry(
        atoms=clean_template.copy(),
        miller_index="100"
    )
    #template_atoms_list_new = []
    #for template_atoms in template_atoms_list:
    #    atoms = template_atoms.copy()
    #    atoms.cell = None
    #    template_atoms_list_new.append(atoms)
    #atoms_list = read("test_relax.traj", index=":")
    write("inverted_template.png", images=[inverted_slab], rotation='10z,-60x')


if __name__ == "__main__":
    main()