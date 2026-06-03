from chgnet.model.dynamics import CHGNetCalculator
from gen_catalyst_design.stability import relax_atoms_and_cell, apply_inversion_symmetry, get_connectivity, recon_check_from_connectivity
from gen_catalyst_design.utils import get_full_element_pool_no_saas, get_periodic_surface
from ase.data import atomic_numbers, reference_states
from ase_ml_models.yaml import write_to_yaml
from ase.io import write



def main():
    facets = ["100", "111"]
    element_pool = get_full_element_pool_no_saas()
    fmax = 0.05
    calc = CHGNetCalculator()
    connectivity_kwargs = dict(
        method="ase",
        bond_cutoff=1.0,
        remove_pbc=True,
        skin=0.2
    )

    energies_ref = {}
    for facet in facets:
        energies_ref[facet] = {}
        atoms_ref, _ = get_periodic_surface(
                miller_index=facet,
                a_lat=None
        )
        atoms_inv_ref = apply_inversion_symmetry(
                atoms=atoms_ref.copy(),
                miller_index=facet,
                a_lat=None
        )

        init_connectivity = get_connectivity(
            atoms=atoms_inv_ref,
            **connectivity_kwargs
        )
        
        for element in element_pool:
            a_lat = None#reference_states[atomic_numbers[element]]["a"]
            atoms, _ = get_periodic_surface(
                miller_index=facet,
                a_lat=a_lat
            )
            atoms.symbols = [element] * len(atoms)

            atoms_inv = apply_inversion_symmetry(
                atoms=atoms.copy(),
                miller_index=facet,
                a_lat=a_lat
            )
            #init_connectivity = get_connectivity(
            #    atoms=atoms_inv,
            #    **connectivity_kwargs
            #)
            atoms_inv.info["init_connectivity"] = init_connectivity.copy()
            atoms_inv.info["connectivity_kwargs"] = connectivity_kwargs
            atoms_inv.calc = calc
            atoms, _ = relax_atoms_and_cell(
                atoms=atoms_inv,
                fmax=fmax,
                trajectory=f"{facet}_{element}.traj",
                relax_z=False
            )
            recon_check = recon_check_from_connectivity(
                atoms=atoms
            )
            ref_e = atoms.get_potential_energy()/len(atoms)
            print("-----------------------------RESULTS------------------------------------")
            print(f"Element {element} relaxed on facet: {facet}")
            print(f"ref. energy: {ref_e:.3f}")
            print(f"has reconstructed: {recon_check}")
            print("-----------------------------RESULTS------------------------------------")
            #exit()
            energies_ref[facet][element] = {"ref_energy":ref_e, "recon":recon_check}
    
    write_to_yaml(
        "energies_ref_surface.yaml",
        data=energies_ref
    )




if __name__ == "__main__":
    main()