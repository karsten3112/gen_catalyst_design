

from gen_catalyst_design.utils import get_atoms_from_template_db
from gen_catalyst_design.stability import Stabilizer
from chgnet.model.dynamics import CHGNetCalculator
from ase.io import read, write
import random


def main():
    random_seed = 42
    random.seed(random_seed)
    element_pool = ["Rh", "Cu", "Au"]
    miller_index = "100"
    calculator = CHGNetCalculator()
    write_atoms_list = True
    n_samples = 1

    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db",
        pth_header="../databases/surface_templates"
    )

    stabilizer = Stabilizer(
        template_atoms=template_atoms_list[0],
        calculator=calculator,
        ref_energy_file="chgnet_ref_energies.yaml",
        ref_energy_pth_header="../yaml_files/reference_energies"
    )

    atoms_list = []
    for _ in range(n_samples):
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        e_form, atoms = stabilizer.get_formation_energy_from_symbols(symbols=symbols, trajectory="test_relax.traj")
        atoms.info["e_form"] = e_form
        atoms_list.append(atoms)
        print(f"Formation energy of surface: E_form = {e_form:+7.3e} [eV]")
    
    if write_atoms_list:
        write("e_form_test.traj", atoms_list)

    



if __name__ == "__main__":
    main()