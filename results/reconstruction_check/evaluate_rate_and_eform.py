from gen_catalyst_design.optimization import setup_optimization_objective
from gen_catalyst_design.stability import calculate_surface_formation_energy
from ase_ml_models.utilities import get_connectivity
from chgnet.model.dynamics import CHGNetCalculator
from ase.io import read, write
import numpy as np
import yaml
import os



def main():
    miller_index = "100"
    template_type = "surface"
    calculator = CHGNetCalculator()
    connectivity_kwargs = dict(
        method="ase",
        bond_cutoff=1.0,
        remove_pbc=True,
        skin=0.2
    )


    reaction_mechanism, stabilizer, template_atoms_list = setup_optimization_objective(
        miller_index=miller_index,
        template_type=template_type,
        database_pth_header="../../databases",
        yaml_files_header="../../yaml_files",
        include_stability=False
    )

    energies_ref = load_ref_energy_dict(
        filename="chgnet_ref_energies.yaml",
        pth_header="../../yaml_files/reference_energies"
    )


    atoms_list = read("chgnet/au_close_fcc.traj", index=":")
    ordered_atoms_dict = order_init_final(atoms_list=atoms_list)
    recons_list = []
    e_form_list = []
    symbols_list = []
    for sample_num in ordered_atoms_dict:
        atoms_init, atoms_final = ordered_atoms_dict[sample_num][0], ordered_atoms_dict[sample_num][1]
        atoms_final.calc = calculator
        energy = atoms_final.get_potential_energy()
        all_symbols = atoms_final.get_chemical_symbols()
        symbols_list.append(all_symbols[0:36])
        is_reconstructed = reconstruction_check(
            atoms_init=atoms_init,
            atoms_final=atoms_final,
            connectivity_kwargs=connectivity_kwargs
        )
        recons_list.append(is_reconstructed)
        e_form = calculate_surface_formation_energy(
            atoms=atoms_final,
            energies_ref=energies_ref
        )
        e_form_list.append(e_form)
    rate_list = []
    for symbols in symbols_list:
        result_dict = reaction_mechanism.get_rate_of_RDS_from_symbols(
            symbols=symbols
        )
        rate = result_dict["rate"]
        rate_list.append(rate)

    result_atoms_list = []
    for symbols, rate, e_form, is_recon in zip(symbols_list, rate_list, e_form_list, recons_list):
        print(rate, e_form, is_recon)
        atoms = template_atoms_list[0].copy()
        atoms.symbols = symbols
        atoms.info["rate"] = rate
        atoms.info["e_form"] = e_form
        atoms.info["is_recon"] = int(is_recon)
        result_atoms_list.append(atoms)
    
    write("chgnet_result_au_close.traj", result_atoms_list)
    



def load_ref_energy_dict(filename:str, pth_header:str=None):
        if pth_header is not None:
            filename = os.path.join(pth_header, filename)
        with open(filename, mode="r") as fileobj:
            data = yaml.safe_load(fileobj)
        return data


def order_init_final(
        atoms_list:list
    ):
    ordered_atoms_dict = {}
    for atoms in atoms_list:
        index = atoms.info["sample_num"]
        if index in ordered_atoms_dict:
            ordered_atoms_dict[index].append(atoms)
        else:
            ordered_atoms_dict[index] = [atoms]
    return ordered_atoms_dict

def reconstruction_check(
        atoms_init,
        atoms_final,
        connectivity_kwargs:dict={}
    ):
    init_connectivity = get_connectivity(
                atoms=atoms_init,
                **connectivity_kwargs
            )
    final_connectivity = get_connectivity(
                atoms=atoms_final,
                **connectivity_kwargs
            )
    con_diff = init_connectivity - final_connectivity
    mask = np.bool(np.abs(con_diff))
    if True in mask:
        return True
    else:
        return False

if __name__ == "__main__":
    main()