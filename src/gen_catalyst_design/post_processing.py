from sklearn.neighbors import KernelDensity
from .db import Database, load_datadicts_from_db
from scipy.stats import ecdf, iqr
import numpy as np


def get_distribution_dict(
        database:Database,
        n_budget_samples:int=10000,
        num_distributions:int=10,
        seperate_initial_samples:bool=False,
        initial_seperation_budget:int=100,
        use_log:bool=False,
        filter_copies:bool=True
    ):
    datadicts = load_datadicts_from_db(database=database)
    if filter_copies:
        datadicts = filter_identical_structures(datadicts=datadicts)
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
    all_samples_used = False
    while sample_count < n_budget_samples and all_samples_used == False:
        dist = []
        for _ in range(samples_per_dist):
            try:
                dist.append(rates[sample_count])
                sample_count+=1
            except:
                all_samples_used = True
        distribution_dict[distribution_count] = np.log(np.array(dist)) if use_log else np.array(dist)
        distribution_count+=1
    return distribution_dict

def filter_identical_structures(
        datadicts:list
    ):
    filtered_dicts = []
    stored_keys = []
    for datadict in datadicts:
        key = "".join(datadict["elements"])
        if key not in stored_keys:
            stored_keys.append(key)
            filtered_dicts.append(datadict)
    return filtered_dicts


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
    plot_samples = np.linspace(plot_range[0], plot_range[1], 10000)
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


def get_ecdf_sf(
        distribution:np.array
    ):
    result = ecdf(distribution)
    return result.cdf, result.sf

def get_top_k_solutions(
        distribution:np.array,
        top_k:int=10,    
    ):
    indices = np.argpartition(distribution, top_k)[:top_k]
    return distribution[indices], indices

def get_accummax_curve(
        distribution:np.array
    ):
    max_curve = []
    for i, value in enumerate(distribution):
        if i == 0:
            max_curve.append(value)
        else:
            if value > max_curve[-1]:
                max_curve.append(value)
            else:
                max_curve.append(max_curve[-1])
    return np.array(max_curve)

def get_area_under_auc(
        accum_max_curve:np.array,
        normalize:bool=False,
        #add_log_offset:float=1.0
    ):
    if normalize:
        accum_max_curve/=np.max(accum_max_curve)[0]
    return np.sum(np.abs(accum_max_curve))
    
def get_unique_structure_freq(
        elements_list:list
    ):
    tot_samples = len(elements_list)
    stored_keys = []

    #Maybe add rotational invariance into check here

    for elements in elements_list:
        query = "".join(elements)
        if query not in stored_keys:
            stored_keys.append(query)
    num_unique_samples = len(stored_keys)
    return num_unique_samples/tot_samples


def get_tot_summary_dict(
        distribution:np.array,
        top_k:int=100,
        elements_list:list=None,
        use_log:bool=True
    ):
    tot_summary_dict = {}
    if use_log:
        distribution = np.log10(distribution)
    
    tot_summary_dict["max_val"] = np.max(distribution)
    tot_summary_dict["min_val"] = np.min(distribution)

    if elements_list is not None:
        unique_freq = get_unique_structure_freq(
            elements_list=elements_list
        )
        tot_summary_dict["unique_freq"] = unique_freq


    distribution_summary = get_dist_summary(
        distribution=distribution
    )

    tot_summary_dict["whole_distribution"] = distribution_summary

    top_k_solutions, indices = get_top_k_solutions(
        distribution=distribution,
        top_k=top_k
    )

    tot_summary_dict["top_k_solutions"] = top_k_solutions
    tot_summary_dict["top_k_indices"] = indices

    top_k_summary = get_dist_summary(
        distribution=top_k_solutions
    )
    tot_summary_dict["top_k_summary"] = top_k_summary
    return tot_summary_dict


def get_dist_summary(
        distribution:np.array,
    ):
    
    median = np.median(distribution)
    mean = np.mean(distribution)
    IQR = iqr(distribution)

    return {"mean":mean, "median":median, "IQR":IQR}


def get_survival_func(
        distribution:np.array,
        use_log:bool=False
    ):
    if use_log:
        distribution = np.log10(distribution)
    ecdf, sf = get_ecdf_sf(
        distribution=distribution
    )
    return {"sf":sf, "in_log_space":use_log}

def get_accum_max_curve(
        distribution:np.array,
        use_log:bool=False,
        get_auc:bool=True
    ):
    if use_log:
        distribution = np.log10(distribution)
    
    max_curve = []
    for i, value in enumerate(distribution):
        if i == 0:
            max_curve.append(value)
        else:
            if value > max_curve[-1]:
                max_curve.append(value)
            else:
                max_curve.append(max_curve[-1])
    max_curve = np.array(max_curve)
    if get_auc:
        auc = np.sum(np.abs(max_curve))
    else:
        auc = None
    return {"max_curve":max_curve, "auc": auc}


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

        acc_max_dict = get_accum_max_curve(
            distribution=rate_distribution
        )
        accum_max_curve_dicts.append(acc_max_dict)
    
    return summary_dicts, survival_func_dicts, accum_max_curve_dicts