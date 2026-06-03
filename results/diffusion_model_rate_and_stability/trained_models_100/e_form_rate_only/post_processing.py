from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.post_processing import (
    make_scatter_hist_plot, swap_reference_energies
)
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml
import os

def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams['hatch.linewidth'] = 0.5

    miller_index = "100"
    model_name = "model_8_ni_close"
    #samples_dir = "samples_joint"
    dataset_pth = os.path.join("../../..","gen_catalyst_design/results/optimization/stability_and_rate",f"{miller_index}_results","only_fcc_ni_close")
    swap_references = True
    alpha = 0.6
    xlims = None #[2.0, 3.0]
    ylims = None #[-2.2, -1.0]


    title_dict = {
        "model_3_ni_close":"Model-3",
        "model_8_ni_close":"Model-8"
        #"model_set_3":"Model-1",
        #"model_set_1":"Model-5"
    }
    model_color = {
        "model_3_ni_close":"C3",
        "model_8_ni_close":"C8"
    }


    fig = plt.figure(figsize=(16, 5))

    outer = fig.add_gridspec(
        1, 3,
        wspace=0.2   # spacing between the 3 joint plots
    )
    axes = []
    for i in range(3):
        inner = outer[i].subgridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            wspace=0.02,  # tight: scatter <-> histy
            hspace=0.02   # tight: histx <-> scatter
        )
        ax_histx = fig.add_subplot(inner[0, 0])
        ax_histx.set_yticks([])
        ax_histx.set_xticklabels([])
        ax_scatter = fig.add_subplot(inner[1, 0])
        ax_histy   = fig.add_subplot(inner[1, 1])
        ax_histy.set_yticklabels([])
        ax_histy.set_xticks([])
        axes.append({"ax_scatter":ax_scatter, "ax_rate_hist":ax_histx, "ax_eform_hist":ax_histy})

    ref_energies_pth_header = "../../../gen_catalyst_design/yaml_files/reference_energies"
    with open(os.path.join(ref_energies_pth_header, "chgnet_ref_energies.yaml"), "r") as fileobj:
        ref_energies_old = yaml.safe_load(fileobj)

    with open(os.path.join(ref_energies_pth_header, "energies_ref_surface.yaml"), "r") as fileobj:
        ref_energies_new_tot = yaml.safe_load(fileobj)
    ref_energies_new = {element:ref_energies_new_tot[miller_index][element]["ref_energy"] for element in ref_energies_new_tot[miller_index]}


    ref_db = Database.establish_connection(
        filename="rnd_seed_0_samples.db",
        pth_header=dataset_pth
    )
    ref_datadicts = load_datadicts_from_db(database=ref_db)

    if swap_references:
        swap_reference_energies(
            datadicts=ref_datadicts,
            energies_ref_old=ref_energies_old,
            energies_ref_new=ref_energies_new
        )

    datadicts_add_eform = []
    for cond_num in range(5):
        db = Database.establish_connection(
            f"condition_{cond_num+2}.db",
            pth_header=os.path.join("..", model_name, "samples_eform_only")
        )
        datadicts_add_eform+=load_datadicts_from_db(database=db)

    if swap_references:
        swap_reference_energies(
            datadicts=datadicts_add_eform,
            energies_ref_old=ref_energies_old,
            energies_ref_new=ref_energies_new
        )

    datadicts_add_rate = []
    for cond_num in range(5):
        db = Database.establish_connection(
            f"condition_{cond_num+2}.db",
            pth_header=os.path.join("..", model_name, "samples_rate_only")
        )
        datadicts_add_rate+=load_datadicts_from_db(database=db)

    if swap_references:
        swap_reference_energies(
            datadicts=datadicts_add_rate,
            energies_ref_old=ref_energies_old,
            energies_ref_new=ref_energies_new
        )

    for ax, add_dicts, add_colors in zip(
        axes, 
        [[datadicts_add_eform], [datadicts_add_rate], [datadicts_add_eform, datadicts_add_rate]],
        [["C2"], ["C3"], ["C2","C3"]]
        ):
        make_scatter_hist_plot(
            ref_datadicts=ref_datadicts[0:1500],
            datadicts_add=add_dicts,
            datadicts_add_colors=add_colors,
            ax_scatter=ax["ax_scatter"],
            ax_rate_hist=ax["ax_rate_hist"],
            ax_eform_hist=ax["ax_eform_hist"],
            xlims=xlims,
            ylims=ylims,
            alpha=alpha,
            hist_kwargs={"num_bins":50},
            percentile=80.0
        )

    legend_elements = [
        Patch(facecolor="C0", edgecolor="black",
          linewidth=1.0,
          alpha=alpha,
          label=f"GA 0–1.5k"),
        Patch(facecolor="C2", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
        label=rf"$w_g$"+"(R, "+r"$E_{form}$) "+r"$=0.2$" + "\n"
          +r"$w_g$"+r"($\emptyset_R$, "+r"$E_{form}$) "+r"$=2.0$" + "\n"
          +r"$w_g$"+r"(R, "+r"$\emptyset_{E_{form}}$) "+r"$=0.0$"
          ),
        Patch(facecolor="C3", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
        label=rf"$w_g$"+"(R, "+r"$E_{form}$) "+r"$=0.2$" + "\n"
          +r"$w_g$"+r"($\emptyset_R$, "+r"$E_{form}$) "+r"$=0.0$" + "\n"
          +r"$w_g$"+r"(R, "+r"$\emptyset_{E_{form}}$) "+r"$=2.0$"
        )
    ]

    fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=4,
            fontsize=12,
            frameon=True
        )
    
    title_patch = Patch(facecolor=model_color[model_name], edgecolor="black")
    leg2 = axes[1]["ax_scatter"].legend(
        handles=[title_patch],
        labels=[title_dict[model_name]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
        frameon=False,
        handlelength=1,
        handletextpad=0.4,
        prop={'size': 16}
    )
    axes[1]["ax_scatter"].add_artist(leg2)
    #fig.subplots_adjust(top=1.22)
    fig.subplots_adjust(bottom=0.22)

    plt.savefig(f"{model_name}_rate_eform.pdf", bbox_inches="tight", pad_inches=0.22, dpi=200)


if __name__ == "__main__":
    main()