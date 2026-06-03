from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.utils import (
    get_atoms_from_template_db, get_full_element_pool_no_saas, get_element_hatches, get_atom_color_dict
)
from gen_catalyst_design.post_processing import (
    get_rates_and_eforms_from_datadicts, swap_reference_energies, filter_identical_structures
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from ase.io import write
import matplotlib as mpl
import numpy as np
import yaml
import os



def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams['hatch.linewidth'] = 0.5

    model_sets = ["model_8_ni_close", "model_3_ni_close"]
    percentile = 75.0
    use_log = True
    miller_indices = ["100", "111"]
    swap_references = True
    add_extending_lines = True
    add_ref_scatter = True
    add_model_scatters = True
    write_atoms_images = False
    alpha = 0.8
    linewidth = 1.5

    element_pool = ["Cu", "Rh", "Ir", "Pd", "Ni"]

    element_color_dict = get_atom_color_dict(
        element_pool=element_pool
    )
    element_hatch_dict = get_element_hatches(
        element_pool=element_pool
    )

    model_pth_header = "trained_models_no_saas"
    dataset_pth_header = "../../gen_catalyst_design/results/optimization/stability_and_rate"

    xlims_dict = {
        "100":[2.05, 2.9],
        "111":[-1.65, 0.0]
    }
    ylims_dict = {
        "100":[-2.2, -0.95],
        "111":[-1.6, -0.4]
    }

    model_color = {
        "model_8_ni_close":"C8",
        "model_3_ni_close":"C3",
        "model_set_1":"C5",
        "ga_final":"C2",
        "ga_init":"C0"
    }

    model_zorder = {
        "ga_init":5,
        "model_8_ni_close":40,
        "ga_final":20,
        "model_3_ni_close":30
    }

    model_labels = {
        "model_8_ni_close":"Model-8 (1.5-2k)",
        "model_3_ni_close":"Model-3 (1.5-2k)",
        "ga_final":"GA (1.5-2k)",
        "ga_init":"GA (0-1.5k)"
    }


    ref_energies_pth_header = "../../gen_catalyst_design/yaml_files/reference_energies"
    with open(os.path.join(ref_energies_pth_header, "chgnet_ref_energies.yaml"), "r") as fileobj:
        ref_energies_old = yaml.safe_load(fileobj)

    with open(os.path.join(ref_energies_pth_header, "energies_ref_surface.yaml"), "r") as fileobj:
        ref_energies_new_tot = yaml.safe_load(fileobj)


    fig, axs = plt.subplots(1, len(miller_indices), figsize=(12,5))

    for ax, miller_index in zip(axs, miller_indices):
        stoch_fig, axs_stoch = plt.subplots(2,2, figsize=(8,5), sharey=True, gridspec_kw={"wspace":0.08, "hspace":0.08})#2, len(model_labels), figsize=(12,4), sharey=True)
        legend_handles = [
        Patch(
            facecolor=element_color_dict[el], 
            label=el,
            edgecolor="k",
            hatch=element_hatch_dict[el]
        )
        for el in element_pool
        ]
        stoch_place = 0.09
        if miller_index == "100":
            stoch_fig.legend(
                handles=legend_handles,
                #title="Element",
                bbox_to_anchor=(0.15, stoch_place),#0.15),
                loc="upper left",
                ncol=6,# if row_plot else len(elements)
                #borderaxespad=0.0,
            )
        
        text_place = 0.10 if miller_index == "100" else 0.10
        stoch_fig.text(
            0.5,                # x position in figure coords
            text_place,               # y position
            f"fcc-{miller_index}",
            ha='center',
            va='center',
            fontsize=16
        )

        # Add underline spanning across all subplots
        spacing = 0.13 if miller_index == "100" else 0.02
        line = Line2D(
            [0.11, 1.1],         # x start/end in figure coords
            [stoch_place+spacing,stoch_place+spacing],#[text_place-spacing, text_place-spacing],     # y position in figure coords
            transform=fig.transFigure,
            color='black',
            linewidth=1
        )

        stoch_fig.add_artist(line)



        xlims, ylims = xlims_dict[miller_index], ylims_dict[miller_index]
        ax.set_title(f"fcc-{miller_index}", fontdict={"fontsize":18})
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_ylabel(r"$E_{form}$ [eV]")
        if use_log:
            ax.set_xlabel(r"$\log_{10}$(rate)")
        else:
            ax.set_xlabel(r"rate [1/s]")

        ref_energies_new = {element:ref_energies_new_tot[miller_index][element]["ref_energy"] for element in ref_energies_new_tot[miller_index]}
        dataset_pth = os.path.join(dataset_pth_header, f"{miller_index}_results", "only_fcc_ni_close")
        model_set_datadicts = {}
        for model_set in model_sets:
            datadicts = []
            for cond_num in range(5):
                db = Database.establish_connection(
                    f"condition_{cond_num+2}.db",
                    pth_header=os.path.join("..", model_pth_header+f"_{miller_index}", model_set, "samples_joint")
                )
                datadicts+=load_datadicts_from_db(database=db)
            datadicts = filter_identical_structures(
                datadicts=datadicts,
                filter_symmetry_equivalent=False
            )
            print(f"num identical samples for miller_index:{miller_index}, model:{model_set} = {500-len(datadicts)}")
            if swap_references:
                swap_reference_energies(
                    datadicts=datadicts,
                    energies_ref_old=ref_energies_old,
                    energies_ref_new=ref_energies_new
                )
            model_set_datadicts[model_set] = datadicts

        ref_db = Database.establish_connection(
            filename="rnd_seed_0_samples.db",
            pth_header=dataset_pth
        )

        ga_datadicts = load_datadicts_from_db(database=ref_db)
        if swap_references:
            swap_reference_energies(
                datadicts=ga_datadicts,
                energies_ref_old=ref_energies_old,
                energies_ref_new=ref_energies_new
            )

        ref_ga_datadicts = ga_datadicts[:1500]

        ref_rates, ref_e_forms = get_rates_and_eforms_from_datadicts(
            datadicts=ref_ga_datadicts,
            use_log=use_log
        )
        if add_ref_scatter:
            ax.scatter(ref_rates, ref_e_forms, c=model_color["ga_init"], alpha=alpha-0.2, edgecolors="k")
        
        rate_percentile = np.percentile(ref_rates, percentile)
        e_form_percentile = np.percentile(ref_e_forms, 100.0 - percentile)
        model_set_datadicts["ga_final"] = ga_datadicts[1500:2000]
        model_set_datadicts["ga_init"] = ga_datadicts[:1500]
        
        frac_points_above_thr = {}
        for model_set, ax_stoch in zip(model_set_datadicts, axs_stoch.flatten()):
            filtered_datadicts = filter_percentile_datadicts(
                datadicts=model_set_datadicts[model_set],
                rate_percentile=rate_percentile,
                e_form_percentile=e_form_percentile,
                use_log=use_log
            )
            print(f"num samples for miller_index:{miller_index}, model:{model_set} above 80th percentile = {500-len(filtered_datadicts)}")
            frac_points_above_thr[model_set] = len(filtered_datadicts)/len(model_set_datadicts[model_set])
            pareto_frontier_dicts = get_pareto_frontiers(
                datadicts=filtered_datadicts,
                use_log=use_log
            )
            if write_atoms_images:
                write_pareto_frontier_images(
                    frontier_dicts=pareto_frontier_dicts,
                    miller_index=miller_index,
                    model_set=model_set
                )

            plot_element_occs(
                frontier_dicts=pareto_frontier_dicts,
                ax=ax_stoch,
                model_set=model_set,
                miller_index=miller_index
            )

            rates, e_forms = get_rates_and_eforms_from_datadicts(
                datadicts=pareto_frontier_dicts,
                use_log=use_log
            )

            rates_filtered, e_forms_filtered = get_rates_and_eforms_from_datadicts(
                datadicts=filtered_datadicts,
                use_log=use_log
            )
            if add_model_scatters:
                if model_set != "ga_init":
                   ax.scatter(rates_filtered, e_forms_filtered, c=model_color[model_set], alpha=alpha-0.2, edgecolors="k")

            ax.plot(
                rates, 
                e_forms, 
                "o-", 
                c=model_color[model_set], 
                label=model_labels[model_set], 
                zorder=model_zorder[model_set], 
                markeredgecolor="k",
                markersize=10,
                linewidth=linewidth,
                alpha=alpha/alpha
            )

            for i, rate, e_form in zip(range(len(rates)), rates, e_forms):
                #if miller_index == "100" and i in [1,2,3]:
                ax.plot(rate, e_form, "o", markeredgecolor="k", markersize=11, zorder=model_zorder[model_set]+i, c=model_color[model_set])
                ax.text(rate, e_form, f"{i+1}", ha="center", va="center", zorder=model_zorder[model_set]+i, fontsize=10)
                #else:
                #    ax.text(rate, e_form, f"{i+1}", ha="center", va="center", zorder=model_zorder[model_set], fontsize=10)

            if add_extending_lines:
                ax.hlines(y=e_forms[0], xmin=xlims[0],xmax=rates[0], zorder=model_zorder[model_set]-1, colors=model_color[model_set], alpha=alpha/alpha, linewidths=linewidth)
                ax.vlines(x=rates[-1], ymin=e_forms[-1], ymax=ylims[-1], zorder=model_zorder[model_set]-1, colors=model_color[model_set], alpha=alpha/alpha, linewidths=linewidth)
            
        stoch_fig.savefig(f"stochs_{miller_index}.png", bbox_inches="tight", dpi=200, pad_inches=0.18)

    legend_elements = [
        Patch(facecolor="C0", edgecolor="black",
          linewidth=1.0,
          alpha=alpha,
          label=f"GA 0–1.5k"),
        Patch(facecolor="C2", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=f"GA 1.5-2k"),
        Patch(facecolor="C3", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=f"Diffusion 1.5–2k (Model-3)"),
        Patch(facecolor="C8", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=f"Diffusion 1.5–2k (Model-8)")
    ]

    fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=12,
            frameon=True
        )

    fig.savefig("pareto_swap.pdf" if swap_references else "pareto.pdf", bbox_inches="tight")

def filter_percentile_datadicts(
        datadicts:list,
        rate_percentile:float,
        e_form_percentile:float,
        use_log:bool=True
    ):
    rates, e_forms = get_rates_and_eforms_from_datadicts(
        datadicts=datadicts,
        use_log=use_log
    )
    rate_indices = np.argwhere(rates >= rate_percentile).flatten()
    e_form_indices = np.argwhere(e_forms <= e_form_percentile).flatten()
    filtered_dicts = []
    for idx in rate_indices:
        if idx in e_form_indices:
            filtered_dicts.append(datadicts[idx])
    return filtered_dicts


def get_pareto_frontiers(
        datadicts:list,
        use_log:bool=True
    ):
    e_form_sorted_dicts = sorted(datadicts, key=lambda x: x["e_form"], reverse=False)
    pareto_frontiers = []
    best_e_form = 0.0
    best_rate = 0.0
    for i, datadict in enumerate(e_form_sorted_dicts):
        e_form, rate = datadict["e_form"], datadict["rate"]
        if i == 0:
            best_e_form = e_form
            best_rate = rate
            pareto_frontiers.append(datadict)
        else:
            if e_form >= best_e_form and rate >= best_rate:
                best_e_form = e_form
                best_rate = rate
                pareto_frontiers.append(datadict)
    return pareto_frontiers


def write_pareto_frontier_images(
        frontier_dicts:list,
        miller_index:str,
        model_set:str
    ):

    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
            db_filename=f"{miller_index}_templates.db", 
            pth_header=os.path.join("../../gen_catalyst_design/databases", "surface_templates")
    )

    outdir = os.path.join(miller_index, model_set)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    for i, frontier_dict in enumerate(frontier_dicts):
        atoms = template_atoms_list[0].copy()
        atoms.symbols = frontier_dict["elements"]

        for j, view in enumerate([dict(), dict(rotation='10z,-75x'),dict(rotation='190z,-75x')]):
            write(
                filename=os.path.join(outdir, f"{i}_view_{j}.png"),
                images=[atoms],
                show_unit_cell=False,
                **view
            )

def plot_element_occs(
        frontier_dicts:list,
        ax:object,
        model_set:str,
        miller_index:str
    ):
    ax.set_xlim([0,36])
    ax.set_frame_on(False)
    element_pool = get_full_element_pool_no_saas(
    )
    ax.set_yticks([])
    ax.set_xticks([])

    element_color_dict = get_atom_color_dict(
        element_pool=element_pool
    )
    element_hatch_dict = get_element_hatches(
        element_pool=element_pool
    )
    
    if miller_index == "111":
        active_site_span = [27,31]
    if miller_index == "100":
        active_site_span = [27, 31]

    title_dict = {
        "model_3_ni_close":"Model-3",
        "model_8_ni_close":"Model-8",
        "ga_init":"GA (0-1.5k)",
        "ga_final":"GA (1.5k-2k)"
    }

    model_color = {
        "model_8_ni_close":"C8",
        "model_3_ni_close":"C3",
        "model_set_1":"C5",
        "ga_final":"C2",
        "ga_init":"C0"
    }


    for i, frontier_dict in enumerate(frontier_dicts):
        elements = frontier_dict["elements"]
        for site, element in enumerate(elements):
            ax.fill_between(
                x=[site, site+1], 
                y1=[-i, -i], 
                y2=[-(i+1), -(i+1)], 
                color=element_color_dict[element], 
                hatch=element_hatch_dict[element],
                edgecolor="k",
                linewidth=1
            )
        ax.text(-1, -i-0.5, f"{i+1}", ha="center", va="center")
    
    color = "C3"

    ax.vlines(
        x=active_site_span,
        ymin=-(i+1)-0.2,
        ymax=0.0+0.2,
        colors=color,
        linewidths=2,
        zorder=102
    )

    ax.vlines(
        x=[9,18,27],
        ymin=-(i+1)-0.2,
        ymax=0.0+0.2,
        colors="k",
        linewidths=2
    )

    ax.hlines(
        y=[0.0+0.2, -(i+1)-0.2],
        xmin=active_site_span[0],
        xmax=active_site_span[1],
        colors=color,
        linewidths=2,
        zorder=100
    )

    #ax.fill_between(
    #    x=active_site_span,
    #    y1=[0+0.2,0+0.2],
    #    y2=[-(i+1)-0.2, -(i+1)-0.2],
    #    color=color,
    #    alpha=0.5,
    #    zorder=100
    #)


    title_patch = Patch(facecolor=model_color[model_set], edgecolor="black", alpha=0.8)
    leg2 = ax.legend(
        handles=[title_patch],
        labels=[title_dict[model_set]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        frameon=False,
        handlelength=1,
        handletextpad=0.4,
        prop={'size': 16}
    )
    ax.add_artist(leg2)

    





if __name__ == "__main__":
    main()