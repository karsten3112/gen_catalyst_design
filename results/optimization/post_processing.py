from gen_catalyst_design.post_processing import get_search_statistics
import matplotlib.pyplot as plt
import numpy as np
import os



def main():
    fig, axs = plt.subplots(1,2, layout="constrained", figsize=(10,4))

    search_method_dicts = {
        "random_search":{"pth_header":"results", "db_filenames":[f"rnd_{i}_seed.db" for i in range(5)]},
        "genetic_algorithm":{"pth_header":None, "db_filenames":["test_opt.db"]},
        "annealing":{"pth_header":None, "db_filenames":["default.db"]}
    }

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
        print([summary_dict["unique_freq"] for summary_dict in summary_dicts])
        #print(accum_max_curve_dicts)
        get_sf_mean_plot(
            ax=axs[0],
            survival_func_dicts=survival_func_dicts,
            summary_dicts=summary_dicts,
            add_all_traj=False,
            use_log=True
        )

        get_mean_accum_max_curve(
            ax=axs[1],
            acc_max_curve_dicts=accum_max_curve_dicts,
            use_log=True
        )

    plt.savefig("result.png")


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

    rate_evals = np.logspace(glob_min_rate, glob_max_rate, 10000)#np.linspace(glob_min_rate, glob_max_rate, 10000)
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
        acc_max_curve_dicts:list,
        add_all_traj:bool=False,
        use_log:bool=True,
        plot_kwargs:dict={}
    ):

    accum_max_curves = np.array([accmax_dict["max_curve"] for accmax_dict in acc_max_curve_dicts])
    mean_max_curve = np.mean(accum_max_curves, axis=0)
    #n_samples_accum = np.logspace(np.log10(1), np.log10(len(mean_max_curve)+1), len(mean_max_curve))
    n_samples_accum = np.arange(1, len(mean_max_curve)+1, 1)
    if add_all_traj:
        for accum_max_curve in accum_max_curves:
            ax.step(n_samples_accum, accum_max_curve, "C3", alpha=0.4)
    
    ax.step(n_samples_accum, mean_max_curve, "k")
    if use_log:
        ax.set_yscale("log")


if __name__ == "__main__":
    main()