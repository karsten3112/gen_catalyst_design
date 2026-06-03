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
    models = ["model_3_active_finale", "model_8_active_finale"]
    miller_index = "100"
    element_pool = get_full_element_pool_no_saas()
    element_color_dict = get_atom_color_dict(element_pool=element_pool)
    model_color = {
        "model_3_active_finale":"C3", 
        "model_8_active_finale":"C8"
    }
    title_dict = {
        "model_3_active_finale":"Model-3", 
        "model_8_active_finale":"Model-8"
    }

    element_hatch = get_element_hatches(
        element_pool=element_pool
    )

    fig, axs = plt.subplots(1,2, figsize=(10,10))
    xs = np.arange(0,21,5)

    model_dicts = {}

    db = Database.establish_connection(
        filename=f"rnd_seed_10_samples.db",
        pth_header=os.path.join("../../../../gen_catalyst_design/results/optimization/genetic_algorithm/results_saas_fix_100")
    )
    datadicts_ga = load_datadicts_from_db(database=db)
    for datadict in datadicts_ga:
        datadict["model"] = "GA"

    for j, ax, model in zip(range(2), axs, models):
        model_dir = os.path.join("..", model)
        pre_db = Database.establish_connection(
            filename="pre_estim.db",
            pth_header=model_dir
        )
        diff_datadicts = load_datadicts_from_db(pre_db)
        sample_db = Database.establish_connection(
            filename="samples.db",
            pth_header=os.path.join(model_dir, "learn_iter_1")
        )
        diff_datadicts += load_datadicts_from_db(sample_db)

        for datadict in diff_datadicts:
            datadict["model"] = model
        
        datadicts_plot = diff_datadicts + datadicts_ga
        print(len(datadicts_plot))
        #filtered_dicts = datadicts_plot
        filtered_dicts = filter_identical_structures(
            datadicts=datadicts_plot,
            filter_symmetry_equivalent=True,
            miller_index=miller_index
        )

        filtered_dicts = sort_datadicts_rate(
            datadicts=filtered_dicts
        )[:top_k]

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
            if datadict["model"] == "GA":
                ax.hlines(y=top_k-(i+1), xmax=21, xmin=0, colors="red")
                ax.hlines(y=top_k-i, xmax=21, xmin=0, colors="red")
                ax.fill_between(
                    x=[0, 21], 
                    y1=[top_k-i, top_k-i], 
                    y2=[top_k-(i+1), top_k-(i+1)], 
                    color="C3", 
                    alpha=0.3
                )

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
        bbox_to_anchor=(0.15, 0.06),
        loc="upper left",
        ncol=6,# if row_plot else len(elements)
        #borderaxespad=0.0,
    )


    plt.savefig(f"stochs.png", bbox_inches="tight", dpi=300, pad_inches=0.175)


def sort_datadicts_rate(
        datadicts:list,
        order:str="desc"
    ):
    sorted_dicts = sorted(datadicts, key=lambda x: x["rate"], reverse=True if order=="desc" else False)
    return sorted_dicts



if __name__ == "__main__":
    main()