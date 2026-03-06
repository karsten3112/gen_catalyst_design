
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
    perform_relaxation = True
    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db",
        pth_header="../../databases/surface_templates"
    )

    placeholder_tokens = random.choices(population=element_pool, k=n_atoms_surf)

    stabilizer = Stabilizer(
        template_atoms=template_atoms_list[0],
        calculator=calculator,
        ref_energy_file="chgnet_ref_energies.yaml",
        ref_energy_pth_header="../../yaml_files/reference_energies"
    )
    if perform_relaxation:
        atoms_list = []
        vacuums = [2.0, 5,0, 10.0]#5.0,7.5,10.0,12.5,15.0,17.5,20.0]
        for vacuum in vacuums:
            stabilizer.vacuum = vacuum
            e_form, atoms = stabilizer.get_formation_energy_from_symbols(
                symbols=placeholder_tokens,
                trajectory=f"vacuum_{vacuum}.traj"
            )
            atoms.info["vacuum"] = vacuum
            atoms.info["e_form"] = e_form
            atoms_list.append(atoms)
            print(f"Formation energy of surface: {e_form}")
        write(filename="test.traj", images=atoms_list)
    else:
        atoms_list = read(filename="test.traj", index=":")

    



if __name__ == "__main__":
    main()