from ase.io import read, write
from gen_catalyst_design.db import Database, load_datadicts_from_db
from ase_ml_models.utilities import get_connectivity
import numpy as np
import os



def main():
    results_dir = "../100_results/full_element_pool_no_saas"
    rnd_seed = 0
    num_structs = 20
    connectivity_kwargs = dict(
        method="ase",
        bond_cutoff=1.0,
        remove_pbc=True,
        skin=0.2
    )

    atoms_list = read(
        filename=os.path.join(results_dir, f"rnd_seed_{rnd_seed}_samples.traj"),
        index=":"
    )
    ordered_atoms_list_dict = order_init_final(atoms_list=atoms_list)
    not_recons_list = []
    recons_list = []
    for i in range(len(ordered_atoms_list_dict)):
        atoms_init, atoms_final = ordered_atoms_list_dict[i][0], ordered_atoms_list_dict[i][1]
        is_reconstructed = reconstruction_check(
            atoms_init=atoms_init,
            atoms_final=atoms_final,
            connectivity_kwargs=connectivity_kwargs
        )
        if is_reconstructed and len(recons_list) < num_structs:
            recons_list.append([atoms_init, atoms_final])
        elif is_reconstructed == False and len(not_recons_list) < num_structs:
            not_recons_list.append([atoms_init, atoms_final])
        else:
            pass
    
    for i, recon_structs in enumerate(recons_list):
        for state, struct in zip(["init", "final"], recon_structs):
            write(
                filename=f"recon_{i}_{state}.png",
                images=struct,
                **dict(rotation='10z,-75x')
            )
        
    for i, no_recon_structs in enumerate(not_recons_list):
        for state, struct in zip(["init", "final"], no_recon_structs):
            write(
                filename=f"no_recon_{i}_{state}.png",
                images=struct,
                **dict(rotation='10z,-75x')
            )
    
    print(len(recons_list))
    print(len(not_recons_list))





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
    

def get_rates_e_form(
        results_dir:str,
        rnd_seed:str,
        use_log:bool=True,
        connectivity_kwargs:dict={}
    ):
    db = Database.establish_connection(
        filename=f"rnd_seed_{rnd_seed}_samples.db",
        pth_header=results_dir
    )
    datadicts = load_datadicts_from_db(database=db)
    rates = np.array([datadict["rate"] for datadict in datadicts])
    if use_log:
        rates = np.log10(rates)
    e_forms = np.array([datadict["e_form"] for datadict in datadicts])

    atoms_list = read(
        filename=os.path.join(results_dir, f"rnd_seed_{rnd_seed}_samples.traj"),
        index=":"
    )

    ordered_atoms_list_dict = order_init_final(atoms_list=atoms_list)
    colors = []
    not_reconstructed_atoms = []
    for i in range(len(ordered_atoms_list_dict)):
        atoms_init, atoms_final = ordered_atoms_list_dict[i][0], ordered_atoms_list_dict[i][1]
        is_reconstructed = reconstruction_check(
            atoms_init=atoms_init,
            atoms_final=atoms_final,
            connectivity_kwargs=connectivity_kwargs
        )
        if is_reconstructed == False:
            not_reconstructed_atoms+=[atoms_init, atoms_final]
        colors.append(f"C{int(is_reconstructed)}")
    return rates, e_forms, colors, not_reconstructed_atoms



if __name__ == "__main__":
    main()