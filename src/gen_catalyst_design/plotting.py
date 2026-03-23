from sklearn.neighbors import KernelDensity
from .db import Database, load_datadicts_from_db
import numpy as np


def get_distribution_dict(
        database:Database,
        n_budget_samples:int=10000,
        num_distributions:int=10,
        seperate_initial_samples:bool=False,
        initial_seperation_budget:int=100,
        use_log:bool=False,
    ):
    datadicts = load_datadicts_from_db(database=database)
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
    return distribution_dict


def plot_kde_dist(
        ax:object,
        rate_distribution:np.array,
        position:float,
        bandwidth:float=100.0,
        align:str="yaxis",
        kde_kwargs:dict={},
        plot_range:list=[0.0, 12000],
        scale_factor:float=None,
        plot_kwargs:dict={},
        edge_kwargs:dict=None
    ):

    kde = KernelDensity(bandwidth=bandwidth, **kde_kwargs).fit(rate_distribution.reshape(-1,1))
    plot_samples = np.linspace(plot_range[0], plot_range[1], 1000)
    log_dist = kde.score_samples(plot_samples.reshape(-1,1))
    norm_dist = np.exp(log_dist)
    scale_factor = 1.0/np.max(norm_dist) if scale_factor is None else scale_factor
    plot_dist = norm_dist*scale_factor
    if align == "yaxis":
        ax.fill_betweenx(plot_samples, plot_dist + position, position, **plot_kwargs)
        if edge_kwargs is not None:
            ax.plot(plot_dist + position, plot_samples, **edge_kwargs)
    elif align == "xaxis":
        ax.fill_between(plot_samples, plot_dist + position, position, **plot_kwargs)
        if edge_kwargs is not None:
            ax.plot(plot_samples, plot_dist + position, **edge_kwargs)
    else:
        raise Exception("align is not defined")



