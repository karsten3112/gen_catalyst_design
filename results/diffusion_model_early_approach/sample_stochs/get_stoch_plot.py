from gen_catalyst_design.post_processing import filter_identical_structures
from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.utils import get_full_element_pool_no_saas, get_atom_color_dict, get_element_hatches
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12

    top_k = 50
    model = "model_8_early_active" #= ["model_3_active_finale", "model_8_active_finale"]
    miller_index = "100"
    element_pool = get_full_element_pool_no_saas()
    element_color_dict = get_atom_color_dict(element_pool=element_pool)
    model_color = {
        "model_8_early_active":"C8",
        "model_3_active_finale":"C3", 
        "model_8_active_finale":"C8",
        "GA":"C0",
        "GA_init":"C1"
    }
    title_dict = {
        "model_3_active_finale":"Model-3", 
        "model_8_active_finale":"Model-8",
        "model_8_early_active":"Model-8: act. learn. (8-10k)",
        "GA":"GA (0-10k)",
        "GA_init":"GA (0-2k)"
    }

    element_hatch = get_element_hatches(
        element_pool=element_pool
    )

    fig, axs = plt.subplots(1,3,figsize=(15,10))
    xs = np.arange(0,21,5)

    model_dicts = {}

    db = Database.establish_connection(
        filename=f"rnd_seed_10_samples.db",
        pth_header=os.path.join("../../../../gen_catalyst_design/results/optimization/genetic_algorithm/results_saas_fix_100")
    )
    datadicts_ga = load_datadicts_from_db(database=db)
    initial_ga_dicts = datadicts_ga[:2000]

    for datadict in datadicts_ga:
        datadict["model"] = "GA"

    model_dir = os.path.join("..", "model_8_2k_active_finale")
    pre_db = Database.establish_connection(
        filename="pre_estim.db",
        pth_header=model_dir
    )
    diff_datadicts = load_datadicts_from_db(pre_db)
    for learn_iter in [f"learn_iter_{i+1}" for i in range(7)]:
        sample_db = Database.establish_connection(
            filename="samples.db",
            pth_header=os.path.join(model_dir, learn_iter)
        )
        diff_datadicts += load_datadicts_from_db(sample_db)
    
    diff_datadicts_filtered = filter_identical_structures(
        datadicts=diff_datadicts,
        filter_symmetry_equivalent=True,
        miller_index=miller_index
    )
    sorted_diff_dicts = sort_datadicts_rate(
        datadicts=diff_datadicts_filtered
    )[:top_k]
        
    filtered_dicts_ga = filter_identical_structures(
        datadicts=datadicts_ga,
        filter_symmetry_equivalent=True,
        miller_index=miller_index
    )
    sorted_ga_dicts = sort_datadicts_rate(
        datadicts=filtered_dicts_ga
    )[:top_k]

    filtered_dicts_ga_init = filter_identical_structures(
        datadicts=initial_ga_dicts,
        filter_symmetry_equivalent=True,
        miller_index=miller_index
    )  

    sorted_init_ga = sort_datadicts_rate(
        datadicts=filtered_dicts_ga_init
    )[:top_k]

    for ax, filtered_dicts, model in zip(axs, [sorted_init_ga, sorted_diff_dicts, sorted_ga_dicts,], ["GA_init", "model_8_early_active", "GA"]):
        for i, datadict in enumerate(filtered_dicts):
            elements = datadict["elements"]
            for site, element in enumerate(elements):
                ax.fill_between(
                    x=[site, site+1], 
                    y1=[top_k-i, top_k-i], 
                    y2=[top_k-(i+1), top_k-(i+1)], 
                    color=element_color_dict[element], 
                    hatch=element_hatch[element],
                    edgecolor="k",
                    linewidth=1
                )
            #if datadict["model"] == "GA":
            #    ax.hlines(y=top_k-(i+1), xmax=21, xmin=0, colors="red")
            #    ax.hlines(y=top_k-i, xmax=21, xmin=0, colors="red")
            #    ax.fill_between(
            #        x=[0, 21], 
            #        y1=[top_k-i, top_k-i], 
            #        y2=[top_k-(i+1), top_k-(i+1)], 
            #        color="C3", 
            #        alpha=0.3
            #)

        integers = np.array(list(range(0, top_k, 10))) + 2
        rates = np.flip(np.log10(np.array([filtered_dicts[idx]["rate"] for idx in integers])))
        ax.set_yticks(ticks=integers+0.5, labels=np.round(rates, 2))
        ax.set_ylabel(r"$\log_{10}$(rate)")
        ax.set_xlim([0,21])
        ax.set_xticks(xs+0.5, xs)
        ax.set_ylim([0.0,top_k+0.5])
        ax.set_frame_on(False)
        ax.set_xlabel("Sites")
        title_patch = Patch(facecolor=model_color[model], edgecolor="black")
        leg2 = ax.legend(
            handles=[title_patch],
            labels=[title_dict[model]],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.045),
            frameon=False,
            handlelength=1,
            handletextpad=0.4,
            prop={'size': 16}
        )
        ax.add_artist(leg2)


    legend_handles = [
        Patch(
            facecolor=element_color_dict[el], 
            label=el,
            edgecolor="k",
            hatch=element_hatch[el]
        )
        for el in element_pool
    ]

    fig.legend(
        handles=legend_handles,
        #title="Element",
        bbox_to_anchor=(0.28, 0.06),
        loc="upper left",
        ncol=6,# if row_plot else len(elements)
        #borderaxespad=0.0,
    )


    plt.savefig(f"stochs_many_act.png", bbox_inches="tight", dpi=300, pad_inches=0.175)


def sort_datadicts_rate(
        datadicts:list,
        order:str="desc"
    ):
    sorted_dicts = sorted(datadicts, key=lambda x: x["rate"], reverse=True if order=="desc" else False)
    return sorted_dicts



if __name__ == "__main__":
    main()