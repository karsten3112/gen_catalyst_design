from gen_catalyst_design.utils import get_atoms_from_template_db, get_full_element_pool
from gen_catalyst_design.stability import Stabilizer
from chgnet.model.dynamics import CHGNetCalculator
from ase.io import read, write
import random


def main():
    random_seed = 42
    random.seed(random_seed)
    element_pool = get_full_element_pool()#["Rh", "Cu", "Au"]
    miller_index = "100"
    calculator = CHGNetCalculator()
    #write_atoms_list = True
    n_samples = 1

    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db",
        pth_header="../../databases/surface_templates"
    )

    stabilizer = Stabilizer(
        template_atoms=template_atoms_list[0],
        calculator=calculator,
        ref_energy_file="chgnet_ref_energies.yaml",
        ref_energy_pth_header="../../yaml_files/reference_energies",
        interval=1000
    )

    recon_check_kwargs = dict(
                method="ase",
                bond_cutoff=1.0,
                remove_pbc=True,
                skin=0.30)

    atoms_list = []
    for i in range(n_samples):
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        result_dict = stabilizer.get_formation_energy_from_symbols(
            symbols=symbols, 
            trajectory=f"no_elem_swap.traj",
            apply_recon_check=True,
            recon_check_kwargs=dict(connectivity_kwargs=recon_check_kwargs)
        )
        e_form = result_dict["e_form"]
        has_reconstructed = result_dict["recon"]
        #print(result_dict)
        #if has_reconstructed:
        #if has_reconstructed:
        #    write(f"atoms_recon_{i}.traj", images=[result_dict["atoms"])
        
        #if e_form is not None:
        #    print(f"Formation energy of surface: E_form = {e_form:+7.3e} [eV]")
        #else:
        #    print("reconstruction happened")
        #atoms.info["e_form"] = e_form
        #atoms_list.append(atoms)
    
    #if write_atoms_list:
    #    write("e_form_test.traj", atoms_list)

if __name__ == "__main__":
    main()