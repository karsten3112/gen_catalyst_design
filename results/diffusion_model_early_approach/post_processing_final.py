from gen_catalyst_design.post_processing import plot_kde_dist, plot_rate_histogram, filter_identical_structures
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
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    #ax_right.tick_params(labelleft=False)

    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_left.set_ylabel(r"Normalized counts \%")
    ax_right.set_ylabel(r"Active learning steps")
    for ax in [ax_left, ax_right]:
        ax.set_xlabel(r"$\log_{10}$(rate)")
    
    ax_left.set_ylim([0.0, 8.0])

    model_dir = "model_8_2k_active_finale"
    samples_per_hist = 1000
    use_log = True
    pre_estim_samples = True
    normalize = True


    plot_left_bins = np.linspace(1.5, 4.5, 100)
    plot_right_bins = np.linspace(1.8,4.5, 50)

    if pre_estim_samples:
        db = Database.establish_connection(
            filename="pre_estim.db",
            pth_header=model_dir
        )
        pre_datadicts = load_datadicts_from_db(database=db)
    else:
        pass

    plot_kwargs = {
        "alpha":0.80,
        "linewidth":1.0,
        "color":"C1",
        "zorder":0
    }

    search_alg_compare_dir = "../../../gen_catalyst_design/results/optimization/genetic_algorithm/results_saas_fix_100"
    search_alg_filename = "rnd_seed_10_samples.db"

    search_alg_db = Database.establish_connection(
        filename=search_alg_filename,
        pth_header=search_alg_compare_dir
    )
    search_alg_datadicts = load_datadicts_from_db(database=search_alg_db)


    learn_iter_samples = {0:pre_datadicts}
    for filename in os.listdir(model_dir):
        if "learn_iter" in filename:
            learn_iter = int(filename.split("_")[-1])
            samples_db = Database.establish_connection(
                filename="samples.db",
                pth_header=os.path.join(model_dir, filename)
            )
            sampled_datadicts = load_datadicts_from_db(database=samples_db)
            learn_iter_samples[learn_iter] = sampled_datadicts


    for ax, bins in zip([ax_left, ax_right], [plot_left_bins, plot_right_bins]):
        bars, dist_ga0 = plot_rate_histogram(
            ax=ax,
            datadicts=search_alg_datadicts[0:2000],
            bins=bins,
            apply_log=use_log,
            normalize=normalize,
            plot_kwargs=plot_kwargs
        )
    

    plot_kwargs["zorder"] = -1
    plot_kwargs["color"] = "C0"
    bars, dist_ga1 = plot_rate_histogram(
        ax=ax_left,
        datadicts=search_alg_datadicts[2000:],
        bins=plot_left_bins,
        apply_log=use_log,
        bottom=1.0 if normalize else 5.0,
        normalize=normalize,
        plot_kwargs=plot_kwargs
    )

    plot_kwargs["zorder"] = -2
    plot_kwargs["color"] = "C2"
    rate_dist_sampled = []
    for learn_iter in range(8):
        rate_dist_sampled+=learn_iter_samples[learn_iter]
    bars, dist_diff = plot_rate_histogram(
        ax=ax_left,
        datadicts=rate_dist_sampled,
        bins=plot_left_bins,
        apply_log=use_log,
        bottom=2.5 if normalize else 100,
        normalize=normalize,
        plot_kwargs=plot_kwargs
    )


    rates = np.array([datadict["rate"] for datadict in search_alg_datadicts[0:2000]])
    if use_log:
        rates = np.log10(rates)
    
    percentile = np.percentile(rates, 80)
    ax_left.vlines(
        x=percentile, ymin=0.0, ymax=8.0, colors="k", linestyles="--", linewidths=1.0
    )

    num_dist_plot = 1
    active_learn_tick_poses = []
    spacing = 8.0
    for dist_num in range(8):
        plot_kwargs["zorder"] = -num_dist_plot
        plot_kwargs["color"] = "C0"
        plot_rate_histogram(
            ax=ax_right,
            datadicts=search_alg_datadicts[:2000+(dist_num+1)*samples_per_hist],
            bottom=(dist_num+0.5)*spacing if normalize else (dist_num+0.5)*75,
            bins=plot_right_bins,
            apply_log=use_log,
            normalize=normalize,
            plot_kwargs=plot_kwargs
        )
        num_dist_plot+=1 
        tot_dist_samples = []
        for iter in range(dist_num+1):
            tot_dist_samples+=learn_iter_samples[iter]
    #        num_dist_plot+=1
        plot_kwargs["zorder"] = -num_dist_plot
        plot_kwargs["color"] = "C2"
        pos = (dist_num+0.5)*spacing+0.0 if normalize else (dist_num+0.5)*75+20
        active_learn_tick_poses.append(pos)
        plot_rate_histogram(
        ax=ax_right,
        datadicts=tot_dist_samples,
        bottom=pos,
        #extra_weight=dist_num*samples_per_hist+2000,
        bins=plot_right_bins,
        apply_log=use_log,
        normalize=normalize,
        plot_kwargs=plot_kwargs
        )
        num_dist_plot+=1  

    ax_right.set_yticks(
        ticks=active_learn_tick_poses,
        labels=list(range(len(active_learn_tick_poses)))
    )

    legend_elements = [
        Patch(facecolor="C1", edgecolor="black",
          linewidth=1.0,
          alpha=plot_kwargs["alpha"],
          label=f"GA 0–2k ({np.round(np.sum(dist_ga0), 1)}\%)"),
        Patch(facecolor="C0", edgecolor="black",
          linewidth=1.0,
        alpha=plot_kwargs["alpha"],
          label=f"GA 2–10k ({np.round(np.sum(dist_ga1), 1)}\%)"),
        Patch(facecolor="C2", edgecolor="black",
          linewidth=1.0,
        alpha=plot_kwargs["alpha"],
          label=f"Diffusion 2–10k w. active-learn. ({np.round(np.sum(dist_diff), 1)}\%)")]
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=3,
        fontsize=12,
        frameon=True
    )



    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(bottom=0.22)
    plt.savefig("hist_normed.pdf" if normalize else "hist.pdf", bbox_inches="tight")
    plt.savefig("hist_normed.svg" if normalize else "hist.svg", bbox_inches="tight")

    pass


if __name__ == "__main__":
    main()