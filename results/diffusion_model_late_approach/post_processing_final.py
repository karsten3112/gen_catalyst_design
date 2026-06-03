from gen_catalyst_design.post_processing import plot_kde_dist, filter_identical_structures
from ase_ml_models.yaml import write_to_yaml
from gen_catalyst_design.db import Database, load_datadicts_from_db
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
from matplotlib.patches import Patch






def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12


    #Get the data
    search_alg_compare_dir = "../../../gen_catalyst_design/results/optimization/genetic_algorithm/results_saas_fix_100"
    search_alg_filename = "rnd_seed_10_samples.db"

    search_alg_db = Database.establish_connection(
        filename=search_alg_filename,
        pth_header=search_alg_compare_dir
    )
    search_alg_datadicts = load_datadicts_from_db(database=search_alg_db)
    
    model_dir = "model_3"
    hide_legend = False
    conditions = [f"condition_{i}.db" for i in range(5)]
    datadicts_sampled = []
    for i, samples_dir in enumerate(["samples_42_seed", "samples_43_seed"]):
        for condition in conditions:
            db = Database.establish_connection(
                filename=condition,
                pth_header=os.path.join(model_dir, samples_dir)
            )
            datadicts_sampled+=load_datadicts_from_db(database=db)
        filtered_dicts = filter_identical_structures(
            datadicts=search_alg_datadicts[:8000]+datadicts_sampled,
            filter_symmetry_equivalent=True,
            miller_index="100"
        )
        print(len(filtered_dicts))


    filtered_dicts = filter_identical_structures(
        datadicts=search_alg_datadicts[:8000]+datadicts_sampled,
        filter_symmetry_equivalent=True,
        miller_index="100"
    )
    print("out of the box")
    print(len(filtered_dicts))
    #exit()
    active_learning_dir = f"../../../active_learning_no_saas/rate_only/genetic_alg_dataset_no_saas/{model_dir}_active_finale"
    learn_iters = ["learn_iter_1"]
    pre_estim_samples = "pre_estim.db"

    if pre_estim_samples is not None:
        db = Database.establish_connection(
            filename=pre_estim_samples,
            pth_header=active_learning_dir
        )
        active_learn_sampled = load_datadicts_from_db(database=db)
    else:
        active_learn_sampled = []

    for learn_iter in learn_iters:
        db = Database.establish_connection(
            filename="samples.db",
            pth_header=os.path.join(active_learning_dir, learn_iter)
        )
        active_learn_sampled+=load_datadicts_from_db(database=db)

    filtered_dicts = filter_identical_structures(
        datadicts=search_alg_datadicts[:8000]+active_learn_sampled,
        filter_symmetry_equivalent=True,
        miller_index="100"
    )
    print("from active learning")
    print(len(filtered_dicts))

    fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True)
    bins = np.linspace(2.5, 4.5, 100)

    plot_kwargs = {
        "alpha":0.85,
        "linewidth":1.0,
        "color":"C0",
        "zorder":2
    }
    normalize = True
    displace_vert = True

    ax.set_xlabel(r"$\log_{10}$(rate)")
    if normalize:
        ax.set_ylabel(r"Normalized counts \%")
    else:
        ax.set_ylabel(r"Counts [N]")
    bars, dist_ga0 = plot_rate_histogram(
        ax=ax,
        datadicts=search_alg_datadicts[0:8000],
        bottom=0.0,
        bins=bins,
        normalize=normalize,
        plot_kwargs=plot_kwargs
    )
    
    plot_kwargs["color"] = "C1"
    plot_kwargs["zorder"] = 1
    bars, dist_ga1 = plot_rate_histogram(
        ax=ax,
        datadicts=search_alg_datadicts[8000:],
        bottom=1.0 if normalize else 50 if displace_vert else 0,
        bins=bins,
        extra_weight=0.0,#8000,
        normalize=normalize,
        plot_kwargs=plot_kwargs
        #bins=np.linspace(-10,5.6, 100)
    )

    plot_kwargs["color"] = "C2"
    plot_kwargs["zorder"] = 0
    bars, dist_diff = plot_rate_histogram(
        ax=ax,
        datadicts=datadicts_sampled,
        bottom=2.0 if normalize else 70 if displace_vert else 0,
        bins=bins,
        extra_weight=0.0,#8000,
        normalize=normalize,
        plot_kwargs=plot_kwargs
        #bins=np.linspace(-10,5.6, 100)
    )

    plot_kwargs["color"] = "C3"
    plot_kwargs["zorder"] = -1
    if len(active_learn_sampled) == 0:
        pass
    else:
        bars, dist_diff_act = plot_rate_histogram(
            ax=ax,
            datadicts=active_learn_sampled,
            bottom=3.5 if normalize else 120 if displace_vert else 0,
            bins=bins,
            extra_weight=0.0,#8000,
            normalize=normalize,
            plot_kwargs=plot_kwargs
            #bins=np.linspace(-10,5.6, 100)
        )


    legend_elements = [
        Patch(facecolor="C0", edgecolor="black",
          linewidth=1.0,
          alpha=plot_kwargs["alpha"],
          label=f"GA 0–8k ({np.round(np.sum(dist_ga0), 1)}\%)"),
        Patch(facecolor="C1", edgecolor="black",
          linewidth=1.0,
        alpha=plot_kwargs["alpha"],
          label=f"GA 8–10k ({np.round(np.sum(dist_ga1), 1)}\%)"),
        Patch(facecolor="C2", edgecolor="black",
          linewidth=1.0,
        alpha=plot_kwargs["alpha"],
          label=f"Diffusion 8–10k ({np.round(np.sum(dist_diff), 1)}\%)"),
        Patch(facecolor="C3", edgecolor="black",
          linewidth=1.0,
        alpha=plot_kwargs["alpha"],
          label=f"Diffusion + act.-learn 8–10k ({np.round(np.sum(dist_diff_act), 1)}\%)")
    ]
    if hide_legend:
        pass
    else:
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=4,
            fontsize=12,
            frameon=True
        )


    rates_gen_alg = np.log10(np.array([datadict["rate"] for datadict in search_alg_datadicts[0:8000]]))
    percentile_marker = np.percentile(rates_gen_alg, 80.0)
    y_lims = [0.0, 9.0] if normalize else [0.0, 170]
    #for ax in axs:
    ax.vlines(x=percentile_marker, ymin=y_lims[0], ymax=y_lims[1], linestyles="--", colors="k", linewidths=1.0)
    ax.set_ylim(y_lims)
    title_dict = {
        #"model_set_2":"Model-6",
        #"model_set_3":"Model-1",
        "model_3":"Model-3",
        "model_8":"Model-8"
        #"model_set_1":"Model-5"
    }
    model_color = {
        "model_3":"C3",
        "model_8":"C8"
        #"model_set_2":"C6",
        #"model_set_3":"C1",
        #"model_set_1":"C5"
    }

    title_patch = Patch(facecolor=model_color[model_dir], edgecolor="black")
    leg2 = ax.legend(
        handles=[title_patch],
        labels=[title_dict[model_dir]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        handlelength=1,
        handletextpad=0.4,
        prop={'size': 16}
    )
    ax.add_artist(leg2)

    #fig.subplots_adjust(wspace=0.02)
    if hide_legend:
        pass
    else:
        fig.subplots_adjust(bottom=0.22)
    #fig.subplots_adjust(top=0.75)
    plt.savefig(f"{model_dir}_hist_normed.pdf" if normalize else "hist.pdf", bbox_inches="tight", pad_inches=0.175)
    plt.savefig(f"{model_dir}_hist_normed.svg" if normalize else "hist.svg", bbox_inches="tight", pad_inches=0.175)




def plot_rate_histogram(
        ax:object,
        datadicts:list,
        bins:np.array=None,
        apply_log:bool=True,
        normalize:bool=True,
        bottom:float=0.0,
        extra_weight:float=0.0,
        plot_kwargs:dict={},
        hist_kwargs:dict={}
    ):
    rate_distribution = np.array([datadict["rate"] for datadict in datadicts])
    if apply_log:
        rate_distribution = np.log10(rate_distribution)
    print(np.percentile(rate_distribution, 90.0))
    norm = len(datadicts)
    rate_min = np.min(rate_distribution)
    rate_max = np.max(rate_distribution)


    if bins is None:
        if apply_log:
            bins = np.linspace(rate_min, rate_max, 100)
        else:
            bins = np.logspace(rate_min, rate_max, 50)
    
    hist, bin_edges = np.histogram(a=rate_distribution, bins=bins)
    hist_normed = hist/(norm+extra_weight)*100.0 if normalize else hist
    #print(np.sum(hist_normed))
    bar_obj = ax.bar(bin_edges[:-1], hist_normed, width=np.diff(bin_edges), bottom=bottom, edgecolor="black", align="edge", **plot_kwargs)
    return bar_obj, hist_normed



if __name__ == "__main__":
    main()