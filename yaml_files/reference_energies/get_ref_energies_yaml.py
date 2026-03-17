from chgnet.model.dynamics import CHGNetCalculator
from gen_catalyst_design.stability import relax_atoms_and_cell
from gen_catalyst_design.utils import get_full_element_pool
from ase.build import bulk
from ase_ml_models.yaml import write_to_yaml



def main():
    element_pool = get_full_element_pool(species_exclude=["Ga", "Mn"])#["Rh", "Cu", "Au", "Ni", "Pd", "Co"]
    fmax = 0.05
    calc = CHGNetCalculator()
    energies_ref = {}
    for element in element_pool:
        print(element)
        atoms = bulk(name=element)
        atoms.calc = calc
        atoms = relax_atoms_and_cell(atoms=atoms, fmax=fmax)
        energy = atoms.get_potential_energy() / len(atoms)
        energies_ref[element] = energy
    
    write_to_yaml("chgnet_ref_energies.yaml", data=energies_ref)

if __name__ == "__main__":
    main()