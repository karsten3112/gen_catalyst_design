import matplotlib.pyplot as plt
from gen_catalyst_design.db import Database, load_datadicts_from_db
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12

    random_seed = 42
    np.random.seed(random_seed)
    dist_type = "violin"
    n_budget = 10000
    seperate_initial_dist = False
    use_log = False
    add_scatter = True

    database = Database.establish_connection(
        filename="test_opt.db"
    )

    distribution_dict = get_distribution_dict(
        database=database,
        n_budget_samples=n_budget,
        num_distributions=8,
        seperate_initial_samples=seperate_initial_dist,
        use_log=use_log
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9,4), gridspec_kw={'width_ratios': [10, 1]}, layout="constrained", sharey=True)
    #fig, ax = plt.subplots(figsize=(9,4), layout="constrained")
    distribution_plot(
        ax=ax0,
        distribution_dict=distribution_dict,
        dist_type=dist_type,
        seperate_initial_samples=seperate_initial_dist,
        use_log=use_log,
        final_tot_dist_ax=ax1,
        add_scatter_plot=add_scatter
    )

    plt.savefig(f"plots/{dist_type}_test.png")




def get_gen_iter_indices(
        n_budget:int=10000,
        num_distributions:int=10,
        gen_iter_interval:int=100,
        seperate_initial_samples:bool=True,
        initial_dist_budget:int=100
    ):

    if seperate_initial_samples:
        #print((n_budget-initial_dist_budget)/num_distributions)
        n_samples_per_dist = int((n_budget-initial_dist_budget)/num_distributions)
        gen_iter_dict = {0:[0]}
    else:
        n_samples_per_dist = int(n_budget/num_distributions)
        gen_iter_dict = {}
    
    gen_iters_per_dist = int(np.ceil(n_samples_per_dist/gen_iter_interval))

    for num_dist in range(num_distributions):
        gen_iters = []
        for i in range(gen_iters_per_dist):
            index = num_dist*gen_iters_per_dist + i
            if seperate_initial_samples:
                index+=1
            gen_iters.append(index)
        if seperate_initial_samples:
            gen_iter_dict[num_dist+1] = gen_iters
        else:
            gen_iter_dict[num_dist] = gen_iters
    return gen_iter_dict


def get_distribution_dict(
        database:Database,
        n_budget_samples:int=10000,
        num_distributions:int=10,
        seperate_initial_samples:bool=True,
        initial_seperation_budget:int=100,
        use_log:bool=False,
    ):
    datadicts = load_datadicts_from_db(database=database)
    gen_iter_list = np.array([datadict["gen_iter"] for datadict in datadicts])
    rates = np.array([datadict["rate"] for datadict in datadicts])

    sample_count = 0
    distribution_count = 0
    distribution_dict = {}
    if seperate_initial_samples:
        init_dist = []
        for _ in range(initial_seperation_budget):
            init_dist.append(rates[sample_count])
            sample_count+=1
        distribution_dict[0] = np.log(np.array(init_dist)) if use_log else np.array(init_dist)
        distribution_count+=1
    
    samples_per_dist = int((n_budget_samples-sample_count)/(num_distributions-distribution_count))
    while sample_count < n_budget_samples:
        dist = []
        for _ in range(samples_per_dist):
            try:
                dist.append(rates[sample_count])
                sample_count+=1
            except:
                break
        distribution_dict[distribution_count] = np.log(np.array(dist)) if use_log else np.array(dist)
        distribution_count+=1
        #print(sample_count)
    return distribution_dict

def distribution_plot(
        ax,
        distribution_dict:dict,
        dist_type:str="violin",
        seperate_initial_samples:bool=False,
        add_scatter_plot:bool=True,
        use_log:bool=False,
        final_tot_dist_ax=None,
        plot_kwargs:dict={}    
    ):
    if use_log:
        ax.set_ylabel("$\log($rate$)$ [1/s]")
    else:
        ax.set_ylabel("rate [1/s]")
    ax.set_xlabel("samples [N]")
    if dist_type == "violin":
        make_violion_plot(
            ax=ax,
            distribution_dict=distribution_dict,
            seperate_initial_samples=seperate_initial_samples,
            add_scatter_plot=add_scatter_plot,
            final_tot_dist_ax=final_tot_dist_ax,
            plot_kwargs=plot_kwargs
        )
    elif dist_type == "box_plot":
        make_box_plot(
            ax=ax,
            distribution_dict=distribution_dict
        )
    elif dist_type == "histogram":
        make_histogram_plot(
            ax=ax,
            distribution_dict=distribution_dict
        )
    else:
        raise NotImplementedError(f"Distribution of type: {dist_type} is not implemented")


def make_histogram_plot(
        ax,
        distribution_dict:dict,
        stepsize:float=500.0,
        hist_height_scale:float=3.0,
        plot_kwargs:dict={}
    ):

    all_rates = np.hstack([distribution_dict[dist] for dist in distribution_dict])
    bins = np.arange(np.min(all_rates), np.max(all_rates) + stepsize, stepsize)
    for dist_num in distribution_dict:
        rates = distribution_dict[dist_num]
        hist, bin_edges = np.histogram(rates, bins=bins)
        hist_norm = hist / hist.sum()*hist_height_scale
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        ax.barh(
            y=bin_centers,
            width=hist_norm,
            height=bin_widths,
            left=dist_num,              # displace each distribution along x by ii
            align="center",
            color=plot_kwargs.pop("color", "red"),
            edgecolor=plot_kwargs.pop("edgecolor", "black"),    # lines around each bin
            linewidth=plot_kwargs.pop("linewidth", 1.0),
            alpha=plot_kwargs.pop("alpha",0.8),
        )


def make_violion_plot(
        ax,
        distribution_dict:dict,
        add_scatter_plot:bool=True,
        seperate_initial_samples:bool=True,
        final_tot_dist_ax=None,
        plot_kwargs:dict={}
    ):
    color = plot_kwargs.pop("color", "C2")
    alpha = plot_kwargs.pop("alpha", 0.6)
    spacing = 6.0
    #positions = np.arange(0.0, )

    datasets = [distribution_dict[dist_num] for dist_num in distribution_dict]
    all_rates = np.hstack(datasets)
    positions = np.linspace(0.0, 2.0*len(distribution_dict), len(distribution_dict))#np.array(list(distribution_dict.keys()))
    delta = np.diff(positions)[0]
    #if add_scatter_plot:
    #    positions-=delta/spacing
    
    parts = ax.violinplot(
        datasets,
        positions=positions-delta/spacing if add_scatter_plot else positions,
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )

    for dist_num, body in zip(distribution_dict, parts["bodies"]):
        if seperate_initial_samples and dist_num == 0:
            col = "C0"
        else:
            col = color
        body.set_facecolor(col)
        body.set_edgecolor("black")
        body.set_alpha(alpha)
    
    for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[part].set_linewidth(1.0)
        parts[part].set_color("black")

    if add_scatter_plot:
        for dist_num, pos in zip(distribution_dict, positions):
            rates = distribution_dict[dist_num]
            xs = pos*np.ones(shape=rates.shape) + np.random.randn(len(rates))*0.1 + delta/spacing
            if seperate_initial_samples and dist_num == 0:
                col = "C0"
            else:
                col = color
            ax.scatter(x=xs, y=rates, c=col, alpha=alpha-0.2, s=10.0, zorder=-1, edgecolors="k")#, zorder=2, edgecolors="k"
    
    labels = []
    samples_acc_count = 0
    for dist_num in distribution_dict:
        distribution = distribution_dict[dist_num]
       # print(len(distribution))
        #if seperate_initial_samples and dist_num == 0:
        label = f"{samples_acc_count}-{samples_acc_count+len(distribution)}"
        samples_acc_count+=len(distribution)
        labels.append(label)

    ax.set_xticks(
        ticks=positions,
        labels=labels,
        rotation=20, horizontalalignment='right'
    )

    span = np.max(all_rates) - np.min(all_rates)
    span_percent = 1e-2
    ax.set_ylim([np.min(all_rates)-span*span_percent, np.max(all_rates)+span*span_percent])

    if final_tot_dist_ax is not None:
        parts = final_tot_dist_ax.violinplot(
        [all_rates],
        positions=[0.5],
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(alpha)
        for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
            parts[part].set_linewidth(1.0)
            parts[part].set_color("black")
        final_tot_dist_ax.set_xticks([])
    

def make_box_plot(
        ax,
        distribution_dict:dict,
        add_scatter_plot:bool=True
    ):

    spacing = 1.2
    datasets = [distribution_dict[dist_num] for dist_num in distribution_dict]
    positions = np.linspace(0.0, 15.0, len(distribution_dict))#np.array(list(distribution_dict.keys()))
    if add_scatter_plot:
        positions-=spacing

    ax.boxplot(
        datasets,
        positions=positions,
        widths=0.6,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor="black",
            markeredgecolor="black",
        ),
    )

    if add_scatter_plot:
        for dist_num, pos in zip(distribution_dict, positions):
            rates = distribution_dict[dist_num]
            xs = pos*np.ones(shape=rates.shape) + np.random.randn(len(rates))*0.05 + spacing
            ax.scatter(x=xs, y=rates, c="lightgrey", alpha=0.6, s=5.0, zorder=-1, edgecolors="k")#, zorder=2, edgecolors="k"


def get_rates_from_datadicts(
        datadicts:list,
        use_log:bool=True
    ):
    rates = np.array([datadict["rate"] for datadict in datadicts])
    if use_log:
        return np.log(rates)
    else:
        return rates


if __name__ == "__main__":
    main()