from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.post_processing import get_tot_summary_dict, get_survival_func, get_accum_max_curve
from scipy.stats import iqr
import matplotlib.pyplot as plt
import numpy as np



def main():
    fig, axs = plt.subplots(1,2)
    use_log = True
    results_dir = "results"
    rnd_seeds = list(range(10))

    summary_dicts = []

    survival_funcs = []
    accum_max_curves = []

    for rnd_seed in rnd_seeds:
        db = Database.establish_connection(
            filename=f"rnd_{rnd_seed}_seed.db",
            pth_header=results_dir
        )
        datadicts = load_datadicts_from_db(database=db)
        rate_distribution = np.array([datadict["rate"] for datadict in datadicts])
        elements_list = [datadict["elements"] for datadict in datadicts]
        
        summary_dict = get_tot_summary_dict(
            distribution=rate_distribution,
            elements_list=elements_list,
            use_log=True
        )
        summary_dicts.append(summary_dict)

        sf_dict = get_survival_func(
            distribution=rate_distribution
        )
        survival_funcs.append(sf_dict)

        acc_max, auc = get_accum_max_curve(
            distribution=rate_distribution
        )

        accum_max_curves.append(acc_max)


    get_sf_mean_plot(
        ax=axs[0],
        survival_func_dicts=survival_funcs,
        summary_dicts=summary_dicts,
        add_all_traj=False,
        use_log=True
    )

    get_mean_accum_max_curve(
        ax=axs[1],
        acc_max_curves=accum_max_curves
    )

    plt.savefig("tot.png")


def get_search_statistics(
        database_filenames:list,
        pth_header:str=None
    ):

    summary_dicts = []
    survival_func_dicts = []
    accum_max_curve_dicts = []

    for database_file in database_filenames:
        db = Database.establish_connection(
            filename=database_file,
            pth_header=pth_header
        )
        datadicts = load_datadicts_from_db(database=db)
        rate_distribution = np.array([datadict["rate"] for datadict in datadicts])
        elements_list = [datadict["elements"] for datadict in datadicts]

        summary_dict = get_tot_summary_dict(
            distribution=rate_distribution,
            elements_list=elements_list,
            use_log=True
        )
        summary_dicts.append(summary_dict)

        sf_dict = get_survival_func(
            distribution=rate_distribution
        )
        survival_func_dicts.append(sf_dict)

        acc_max, auc = get_accum_max_curve(
            distribution=rate_distribution
        )
        accum_max_curve_dicts.append(acc_max)
    
    return summary_dicts, survival_func_dicts, accum_max_curve_dicts




def get_sf_mean_plot(
        ax:object,
        survival_func_dicts:list,
        summary_dicts:list,
        add_all_traj:bool=True,
        summary_is_log_space:bool=True,
        use_log:bool=True,
        plot_kwargs:dict={}
    ):

    glob_max_rate = np.max([summary_dict["max_val"] for summary_dict in summary_dicts])
    glob_min_rate = np.min([summary_dict["min_val"] for summary_dict in summary_dicts])
    #if summary_is_log_space:
    #    glob_max_rate = np.exp(glob_max_rate)
    #    glob_min_rate = np.exp(glob_min_rate)

    rate_evals = np.logspace(glob_min_rate, glob_max_rate, 1000)#np.linspace(glob_min_rate, glob_max_rate, 10000)
    probs_arr = np.array([sf_dict["sf"].evaluate(rate_evals) for sf_dict in survival_func_dicts])
    mean_sf = np.mean(probs_arr, axis=0)

    if add_all_traj:
        for prob_arr in probs_arr:
            ax.plot(rate_evals, prob_arr, "C3", alpha=0.4)
    ax.plot(rate_evals, mean_sf, "k", **plot_kwargs)
    if use_log:
        ax.set_xscale("log")


def get_mean_accum_max_curve(
        ax:object,
        acc_max_curves:list,
        add_all_traj:bool=False,
        plot_kwargs:dict={}
    ):

    accum_max_curves = np.array(acc_max_curves)
    mean_max_curve = np.mean(accum_max_curves, axis=0)
    #n_samples_accum = np.logspace(np.log10(1), np.log10(len(mean_max_curve)+1), len(mean_max_curve))
    n_samples_accum = np.arange(1, len(mean_max_curve)+1, 1)
    if add_all_traj:
        for accum_max_curve in accum_max_curves:
            ax.step(n_samples_accum, accum_max_curve, "C3", alpha=0.4)
    
    ax.step(n_samples_accum, mean_max_curve, "k")
    
    ax.set_yscale("log")



if __name__ == "__main__":
    main()