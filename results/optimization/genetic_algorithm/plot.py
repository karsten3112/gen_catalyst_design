from gen_catalyst_design.plotting import get_distribution_dict, plot_kde_dist
from gen_catalyst_design.db import Database
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12

    n_budget = 10000
    seperate_initial_dist = False
    use_log = False
    bandwidth = 200.0
    align="yaxis"

    database = Database.establish_connection(
        filename="test_opt.db"
    )

    distribution_dict = get_distribution_dict(
        database=database,
        n_budget_samples=n_budget,
        num_distributions=10,
        seperate_initial_samples=seperate_initial_dist,
        use_log=use_log
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9,4), gridspec_kw={'width_ratios': [10, 1]}, layout="constrained", sharey=True)

    plot_kwargs = {
        "color":"C3",
        "alpha":0.6,
        "lw":0.0
    }

    edge_kwargs = {
        "color":"k",
        "lw":1.0
    }

    spacing = 0.50
    positions = (np.linspace(0.0, len(distribution_dict), len(distribution_dict))*spacing)

    plot_range = [-1000, 12000]
    #ax0.set_ylim(plot_range)
    #ax0.set_xlim([0.0, np.max(positions)+1.0])
    i = 0
    for dist_num, position in zip(distribution_dict, positions):
        rate_distribution = distribution_dict[dist_num]
        plot_kde_dist(
            ax=ax0,
            bandwidth=bandwidth,
            rate_distribution=rate_distribution,
            position=position,
            plot_kwargs=plot_kwargs,
            edge_kwargs=edge_kwargs,
            plot_range=plot_range,
            align=align
        )
        #if i == 4:
        #    break
        #i+=1
    plt.savefig("test.png")
    exit()
    database = Database.establish_connection(
        filename="rate_9000.0_evals.db",
        pth_header="../../../tests/diffusion_model/full_surface/full_model_test_2/samples"
    )

    distribution_dict = get_distribution_dict(
        database=database,
        n_budget_samples=100,
        num_distributions=1,
        seperate_initial_samples=seperate_initial_dist,
        use_log=use_log,
    )

    plot_kwargs = {
        "color":"C0",
        "alpha":0.6,
        "lw":0.0
    }

    plot_kde_dist(
        ax=ax0,
        bandwidth=bandwidth,
        rate_distribution=distribution_dict[0],
        position=positions[5],
        plot_kwargs=plot_kwargs,
        edge_kwargs=edge_kwargs,
        plot_range=plot_range,
        align=align
    )
    
    plt.savefig("test.png")



if __name__ == "__main__":
    main()