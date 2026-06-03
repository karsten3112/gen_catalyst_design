from gen_catalyst_design.utils import get_atoms_from_template_db, get_periodic_surface
from gen_catalyst_design.stability import apply_inversion_symmetry, get_connectivity
from gen_catalyst_design.db import Database, load_datadicts_from_db
from chgnet.model.dynamics import CHGNetCalculator
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib as mpl
from ase.io import read
import numpy as np
import yaml
import os




def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12

    element_pools = {
        "full_element_pool_no_saas":r"Full element pool",
        "only_fcc":r"Only fcc",
        "only_fcc_au_close": r"Only fcc"+r", $a\sim a_{Au}$",
        "only_fcc_ni_close": r"Only fcc"+r", $a\sim a_{Ni}$"
    }

    rnd_seed = 0
    miller_index = "100"
    use_log = True
    connectivity_kwargs = dict(
        method="ase",
        bond_cutoff=1.0,
        remove_pbc=True,
        skin=0.2
    )
    alpha = 0.6

    #rate_weights = [0.9, 0.7, 0.5, 0.3, 0.1]
    #eform_weights = [0.1, 0.3, 0.5, 0.7, 0.9]

    atoms_ref, _ = get_periodic_surface(
        miller_index=miller_index,
        a_lat=None
    )
    atoms_inv_ref = apply_inversion_symmetry(
            atoms=atoms_ref.copy(),
            miller_index=miller_index,
            a_lat=None
    )
    init_connectivity = get_connectivity(
        atoms=atoms_inv_ref,
        **connectivity_kwargs
    )

    ref_energies_pth_header = "../../../../../yaml_files/reference_energies"

    with open(os.path.join(ref_energies_pth_header, "chgnet_ref_energies.yaml"), "r") as fileobj:
        ref_energies_old = yaml.safe_load(fileobj)


    with open(os.path.join(ref_energies_pth_header, "energies_ref_surface.yaml"), "r") as fileobj:
        ref_energies_new_tot = yaml.safe_load(fileobj)

    ref_energies_new = {element:ref_energies_new_tot[miller_index][element]["ref_energy"] for element in ref_energies_new_tot[miller_index]}


    #calc = CHGNetCalculator()

    tot_reconstruction_dict = {}

    fig, axs = plt.subplots(1,len(element_pools), figsize=(12,4), sharex=True, sharey=True)
    i = 0
    for element_pool, ax in zip(element_pools, axs):
        db = Database.establish_connection(
            filename=f"rnd_seed_{rnd_seed}_samples.db",
            pth_header=os.path.join("..", element_pool)
        )
        datadicts = load_datadicts_from_db(database=db)
        swap_reference_energies(
            datadicts=datadicts,
            energies_ref_old=ref_energies_old,
            energies_ref_new=ref_energies_new
        )

        atoms_traj_list = read(
            filename=os.path.join("..", element_pool, f"rnd_seed_{rnd_seed}_samples.traj"),
            index=":"
        )

        ordered_atoms_dict = order_init_final(
            atoms_list=atoms_traj_list
        )
        reconstructed_list = []
        for sample_idx in ordered_atoms_dict:
            init_atoms, final_atoms = ordered_atoms_dict[sample_idx][0], ordered_atoms_dict[sample_idx][1]
            has_reconstructed = reconstruction_check(
                init_connectivity=init_connectivity,
                atoms_final=final_atoms,
                connectivity_kwargs=connectivity_kwargs
            )
            reconstructed_list.append(has_reconstructed)
        
        tot_reconstruction_dict[f"rnd_seed_{rnd_seed}"] = reconstructed_list
        

        rates, e_forms = get_rates_and_eform(
            datadicts=datadicts,
            use_log=use_log
        )

        ax.scatter(
            rates,
            e_forms,
            c=[f"C{int(recon)}" for recon in reconstructed_list],
            alpha=alpha,
            edgecolor="k"
        )

        if use_log:
            ax.set_xlabel(r"$\log_{10}$(rate)")
        else:
            ax.set_xlabel(r"rate [1/s]")

        ax.set_title(element_pools[element_pool])
        if i == 0:
            ax.set_ylabel(r"$E_{form}$ [eV]")
        i+=1

    legend_elements = [
        Patch(facecolor="C0", edgecolor="black",
          linewidth=1.0,
          alpha=alpha,
          label=f"Not reconstructed"),
        Patch(facecolor="C1", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=f"Reconstructed"),
    ]

    fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=4,
            fontsize=12,
            frameon=True
        )
    
    fig.subplots_adjust(bottom=0.22)
    
    plt.savefig("across_elem_pools.pdf", bbox_inches="tight", dpi=200)


def get_eforms(
        datadicts:list
    ):
    return np.array([datadict["e_form"] for datadict in datadicts])

def get_rates(
        datadicts:list,
        use_log:bool=True
    ):
    rates = np.array([datadict["rate"] for datadict in datadicts])
    if use_log:
        rates = np.log10(rates)
    return rates

def get_rates_and_eform(
        datadicts:list,
        use_log:bool
    ):
    e_forms = get_eforms(
        datadicts=datadicts
    )
    rates = get_rates(
        datadicts=datadicts,
        use_log=use_log
    )
    return rates, e_forms


def swap_reference_energy(
        e_form_old:float,
        symbols:list,
        energies_ref_old:dict,
        energies_ref_new:dict,
    ):
    e_form = e_form_old
    stoichiometries = dict(Counter(symbols))

    old_offset = sum(stoichiometries[ee] * energies_ref_old[ee] for ee in stoichiometries)
    e_form += old_offset
    new_offset = sum(stoichiometries[ee] * energies_ref_new[ee] for ee in stoichiometries)
    e_form -= new_offset
    return e_form

def swap_reference_energies(
        datadicts:list,
        energies_ref_old:dict,
        energies_ref_new:dict,
    ):
    for datadict in datadicts:
        symbols = datadict["elements"]
        e_form = datadict["e_form"]
        e_form_new = swap_reference_energy(
            e_form_old=e_form,
            symbols=symbols,
            energies_ref_old=energies_ref_old,
            energies_ref_new=energies_ref_new
        )
        datadict["e_form"] = e_form_new

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
        init_connectivity,
        atoms_final,
        connectivity_kwargs:dict={}
    ):

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