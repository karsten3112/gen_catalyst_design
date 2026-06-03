from gen_catalyst_design.post_processing import get_search_statistics, plot_kde_dist, plot_rate_histogram
from ase_ml_models.yaml import write_to_yaml
from gen_catalyst_design.db import Database, load_datadicts_from_db
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import matplotlib.patches as patches



def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    # Define grid: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    # Left plot spans both rows
    ax_left = fig.add_subplot(gs[:, 0])
    # Right plots (stacked)
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    kde_dist_common_kwargs = {
        "alpha":0.75,
        #"edgecolor":"k",
        #"linewidth":1.0,
    }
    kde_dist_edge_kwargs = {
        "color":"k",
        "linewidth":1.0
    }


    plot_kwargs_method = {
        "random_search":{
            "color":"C0",
            "linewidth":1.0
        },
        "genetic_algorithm":{
            "color":"C1",
            "linewidth":1.0
        },
        "annealing":{
            "color":"C2",
            "linewidth":1.0
        }
    }

    search_labels = {
        "genetic_algorithm": "Genetic Algorithm",
        "annealing": "Annealing",
        "random_search": "Random Search"
    }

    calculate_avg_stats = True
    plot_kde_dists = True
    rnd_seed_offset = 10
    rnd_seeds = 50
    filenames = [f"rnd_seed_{i+rnd_seed_offset}_samples.db" for i in range(rnd_seeds)]
    filenames_rnd_search = [f"rnd_{i}_seed.db" for i in range(rnd_seeds)]
    search_method_dicts = {
        "genetic_algorithm":{"pth_header":"results_saas_fix_100", "db_filenames":filenames},
        "annealing":{"pth_header":"results_saas_fix_100", "db_filenames":filenames},
        "random_search":{"pth_header":"results_saas_fix_100", "db_filenames":filenames_rnd_search},
    }

    if calculate_avg_stats:
        method_statistics = {}
        for method in search_method_dicts:
            search_dict = search_method_dicts[method]
            if search_dict["pth_header"] is not None:
                results_dir = os.path.join(method, search_dict["pth_header"])
            else:
                results_dir = method
            summary_dicts, survival_func_dicts, accum_max_curve_dicts = get_search_statistics(
                database_filenames=search_dict["db_filenames"],
                pth_header=results_dir
            )

            method_statistics[method] = get_avg_statistics(
                summary_dicts=summary_dicts
            )

            get_sf_mean_plot(
                ax=ax_bottom_right,
                survival_func_dicts=survival_func_dicts,
                summary_dicts=summary_dicts,
                add_all_traj=False,
                use_log=True,
                plot_kwargs=plot_kwargs_method[method]
            )

            get_mean_accum_max_curve(
                ax=ax_top_right,
                acc_max_curve_dicts=accum_max_curve_dicts,
                use_log=True,
                plot_kwargs=plot_kwargs_method[method]
            )

        write_to_yaml(
            filename="result_stats.yaml", data=method_statistics
        )

    num_rnd_seeds = 3
    tot_height = 0.8*(len(search_method_dicts)+num_rnd_seeds)
    if plot_kde_dists:
        ax_left.set_xlabel(r"$\log_{10}($rate$)$")
        ax_left.set_yticks([])
        for i, method in enumerate(search_method_dicts):
            filenames = search_method_dicts[method]["db_filenames"][0:num_rnd_seeds]
            
            if search_method_dicts[method]["pth_header"] is not None:
                results_dir = os.path.join(method, search_method_dicts[method]["pth_header"])

            for j, filename in enumerate(filenames):
                db = Database.establish_connection(
                    filename=filename,
                    pth_header=results_dir
                )
                datadicts = load_datadicts_from_db(database=db)
                plot_kwargs = {"alpha":0.7}
                plot_kwargs.update(plot_kwargs_method[method])
                if j == 0:
                    plot_kwargs.update({"label":search_labels[method]})

                plot_rate_histogram(
                    ax=ax_left,
                    datadicts=datadicts,
                    bins=np.linspace(-7.5,5.0, 100),
                    normalize=True,
                    bottom=tot_height - 6.0*(i*num_rnd_seeds+j),
                    plot_kwargs=plot_kwargs
                )
                
                #rate_dist = np.log10(np.array([datadict["rate"] for datadict in datadicts]))
                #joint_plot_kwargs = {"label":search_labels[method]} if j == 0 else {}
                #joint_plot_kwargs.update(kde_dist_common_kwargs)
                #joint_plot_kwargs.update(plot_kwargs_method[method])
                #plot_kde_dist(
                #    ax=ax_left,
                #    bandwidth=0.8,
                #    rate_distribution=rate_dist,
                #    position=tot_height - 0.8*(i*num_rnd_seeds+j),
                #    align="xaxis",
                #    plot_range=[-7.5,7.5],
                #    plot_kwargs=joint_plot_kwargs,
                #    edge_kwargs=kde_dist_edge_kwargs
                #)
                #ax_left.set_xscale("log")
        ax_left.legend()
    plt.savefig("result_summary_2.pdf")


def get_sf_mean_plot(
        ax:object,
        survival_func_dicts:list,
        summary_dicts:list,
        add_all_traj:bool=True,
        summary_is_log_space:bool=True,
        use_log:bool=True,
        plot_kwargs:dict={}
    ):
    ax.minorticks_on()
    ax.set_ylabel(r"$P(R > r)$"+ r" $\%$")
    ax.set_xlim([10**(-4.5), 10**(6.5)])
    glob_max_rate = 6.5#np.max([summary_dict["max_val"] for summary_dict in summary_dicts])
    glob_min_rate = -4.5#np.min([summary_dict["min_val"] for summary_dict in summary_dicts])
    rate_evals = np.logspace(glob_min_rate, glob_max_rate, 10000)#np.linspace(glob_min_rate, glob_max_rate, 10000)
    probs_arr = np.array([sf_dict["sf"].evaluate(rate_evals) for sf_dict in survival_func_dicts])
    mean_sf = 100*np.mean(probs_arr, axis=0)
    if add_all_traj:
        for prob_arr in probs_arr:
            ax.plot(rate_evals, prob_arr, "C3", alpha=0.4)
    ax.plot(rate_evals, mean_sf, **plot_kwargs)
    if use_log:
        ax.set_xscale("log")
        #ax.set_xlabel(r"$\log(rate)$")
        ax.set_xlabel(r"rate [1/s]")
    else:
        ax.set_xlabel(r"rate [1/s]")

def get_mean_accum_max_curve(
        ax:object,
        acc_max_curve_dicts:list,
        add_all_traj:bool=False,
        use_log:bool=True,
        plot_kwargs:dict={}
    ):
    ax.minorticks_on()
    ax.set_xlabel(r"samples [N]")
    accum_max_curves = np.array([accmax_dict["max_curve"] for accmax_dict in acc_max_curve_dicts])
    mean_max_curve = np.mean(accum_max_curves, axis=0)
    n_samples_accum = np.arange(1, len(mean_max_curve)+1, 1)
    if add_all_traj:
        for accum_max_curve in accum_max_curves:
            ax.step(n_samples_accum, accum_max_curve, "C3", alpha=0.4)
    
    ax.step(n_samples_accum, mean_max_curve, **plot_kwargs)

    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel(r"rate [1/s]")
        #ax.set_ylabel(r"$\log(rate)$")
    else:
        ax.set_ylabel(r"rate [1/s]")

def get_avg_statistics(
        summary_dicts:list,
        stat_measures_dist:list=["mean", "median", "IQR"]
    ):
    statistics_dict = {}
    for dist_type in ["whole_distribution", "top_k_summary"]:
        dist_summary = {}
        for stat_measure in stat_measures_dist:
            values = np.array([summary_dict[dist_type][stat_measure] for summary_dict in summary_dicts])
            stat_dict = get_mean_errs(
                value_arr=values
            )
            dist_summary[stat_measure] = stat_dict
        statistics_dict[dist_type] = dist_summary
        
    for stat_measure in ["max_val", "unique_freq"]:
        values = np.array([summary_dict[stat_measure] for summary_dict in summary_dicts])
        stat_dict = get_mean_errs(
            value_arr=values
        )
        statistics_dict[stat_measure] = stat_dict

    return statistics_dict

def get_mean_errs(
        value_arr:np.array
    ):
    mean = np.mean(value_arr)
    std = np.std(value_arr, ddof=1)
    err = std/np.sqrt(len(value_arr))
    return {"mean":mean, "std":std, "err":err}



if __name__ == "__main__":
    main()