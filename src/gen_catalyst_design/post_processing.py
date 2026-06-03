from sklearn.neighbors import KernelDensity
from collections import Counter
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
        datadicts:list,
        filter_symmetry_equivalent:bool=False,
        miller_index:str="100"
    ):
    unique_samples = {"_".join(datadict["elements"]):datadict for datadict in datadicts}
    if filter_symmetry_equivalent:
        unique_sample_keys = set(unique_samples.keys())
        while len(unique_sample_keys) > 0:
            sample_key = unique_sample_keys.pop()
            symmetry_equiv_structures = get_symmetry_equivalent_surfaces(
                elements=sample_key.split("_"),
                miller_index=miller_index
            )
            for equiv_struct in symmetry_equiv_structures:
                equiv_key = "_".join(equiv_struct)
                if equiv_key in unique_samples:
                    unique_samples.pop(equiv_key)
                    unique_sample_keys.remove(equiv_key)
    return list(unique_samples.values())


def get_symmetry_equivalent_surfaces(
        elements:list,
        miller_index:str="100"  
    ):
    equivalent_surfaces = []
    rotated_elements = elements
    if miller_index == "100":
        for _ in range(3):
            rotated_elements = apply_rotation(
                elements=rotated_elements,
                miller_index=miller_index
            )
            equivalent_surfaces.append(rotated_elements)
    elif miller_index == "111":
        pass
    else:
        raise Exception(f"rotation is not defined for miller-index:{miller_index}")
    
    return equivalent_surfaces

def apply_rotation(
        elements:list,
        miller_index:str="100"
    ):
    idx_rotations_dict = {
        "100":{
            0:1, 1:3, 2:0, 3:2, 4:4, 5:12, 
            6:8, 7:13, 8:10, 9:11, 10:14, 11:20,
            12:18, 13:19, 14:6, 15:7, 16:5, 
            17:9, 18:16, 19:15, 20:17,
        },
        #"111":{

        #}
    }
    if miller_index in idx_rotations_dict:
        rotation_indices = idx_rotations_dict[miller_index]
        rotated_elements = ["(X)"] * len(elements)
        for idx in rotation_indices:
            rotated_elements[rotation_indices[idx]] = elements[idx]
        return rotated_elements
    else:
        raise Exception(f"rotation is not defined for miller-index:{miller_index}")



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
    scale_factor = 1.0#1.0/np.max(norm_dist) if scale_factor is None else scale_factor
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
    indices = np.argpartition(distribution, -top_k)[-top_k:]
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
        get_auc:bool=True,
        num_tot_samples:int=10000,
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
    if len(max_curve) < num_tot_samples:
        diff_samples = num_tot_samples - len(max_curve)
        min_val = max_curve[0]
        max_curve = np.hstack([min_val.repeat(diff_samples), max_curve])
        
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
    #print(np.percentile(rate_distribution, 90.0))
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

def get_rates_from_datadicts(
        datadicts:list,
        use_log:bool=True
    ):
    rates = np.array([datadict["rate"] for datadict in datadicts])
    if use_log:
        rates = np.log10(rates)
    return rates

def get_eforms_from_datadicts(
        datadicts:list
    ):
    e_forms = np.array([datadict["e_form"] for datadict in datadicts])
    return e_forms

def get_rates_and_eforms_from_datadicts(
        datadicts:list,
        use_log:bool=True
    ):
    rates = get_rates_from_datadicts(
        datadicts=datadicts,
        use_log=use_log
    )
    e_forms = get_eforms_from_datadicts(
        datadicts=datadicts
    )
    return rates, e_forms

def plot_histogram(
        ax:object,
        datadicts:list,
        score_key:str,
        align:str="xaxis",
        apply_log:bool=False,
        normalize:bool=True,
        bottom:float=0.0,
        extra_weight:float=0.0,
        plot_kwargs:dict={},
        hist_kwargs:dict={}
    ):
    distribution = np.array([datadict[score_key] for datadict in datadicts])
    if apply_log:
        distribution = np.log10(distribution)
    norm = len(datadicts)
    bins = hist_kwargs["bins"] if "bins" in hist_kwargs else np.linspace(np.min(distribution), np.max(distribution), 100)
    hist, bin_edges = np.histogram(a=distribution, bins=bins)
    hist_normed = hist/(norm+extra_weight)*100.0 if normalize else hist
    if align == "xaxis":
        bar_obj = ax.bar(bin_edges[:-1], hist_normed, width=np.diff(bin_edges), bottom=bottom, edgecolor="black", align="edge", **plot_kwargs, rasterized=True)
    elif align == "yaxis":
        bar_obj = ax.barh(bin_edges[:-1], hist_normed, height=np.diff(bin_edges), edgecolor="black", align="edge", **plot_kwargs, rasterized=True)
    else:
        raise Exception(f"is not possible to plot along: {align}")
    return bar_obj, hist_normed, bins


def plot_scatter_and_hists(
        datadicts:list,
        ax_scatter,
        ax_rate_hist,
        ax_eform_hist,
        use_log:bool=True,
        color:str="C0",
        alpha:float=0.6,
        zorder_scatter:int=0,
        zorder_hist:int=0,
        scatter_kwargs:dict={},
        hist_kwargs:dict={}
    ):
    rates, eforms = get_rates_and_eforms_from_datadicts(
        datadicts=datadicts,
        use_log=use_log
    )
    ax_scatter.scatter(rates, eforms, color=color, edgecolors="k", alpha=alpha, zorder=zorder_scatter, **scatter_kwargs, rasterized=True)
    hist_dict_summary = {}
    for ax, score_key, apply_log, alignment, score_hist_kwargs in zip(
        [ax_rate_hist, ax_eform_hist],
        ["rate", "e_form"],
        [use_log, False],
        ["xaxis", "yaxis"],
        [hist_kwargs["rate"] if "rate" in hist_kwargs else {}, hist_kwargs["e_form"] if "e_form" in hist_kwargs else {}]
        ):
        hist_dict_summary[score_key] = plot_histogram(
            ax=ax,
            datadicts=datadicts,
            score_key=score_key,
            apply_log=apply_log,
            align=alignment,
            plot_kwargs = {
                "alpha":alpha,
                "linewidth":1.0,
                "color":color,
                "zorder":zorder_hist
            },
            hist_kwargs=score_hist_kwargs
        )
    return hist_dict_summary


def make_scatter_hist_plot(
        ref_datadicts:list,
        ax_scatter:object,
        ax_rate_hist:object,
        ax_eform_hist:object,
        datadicts_add:list=None,
        datadicts_add_colors:list=None,
        alpha:float=0.6,
        xlims:list=None,
        ylims:list=None,
        scatter_kwargs:dict={},
        hist_kwargs:dict={},
        percentile:float=None,
        highligt_percentile_region:bool=True,
        use_log:bool=True
    ):
    
    ax_scatter.set_ylabel(r"$E_{form}$ [eV]")
    if use_log:
        ax_scatter.set_xlabel(r"$\log_{10}$(rate)")
    else:
        ax_scatter.set_xlabel(r"rate [1/s]")

    tot_datadicts = ref_datadicts.copy()
    if datadicts_add is not None:
        for added_dicts in datadicts_add:
            tot_datadicts += added_dicts

    rates, e_forms = get_rates_and_eforms_from_datadicts(
        datadicts=tot_datadicts,
        use_log=use_log
    )

    if xlims is None:
        xbuffer = 0.5
        xlims = [np.min(rates)-xbuffer, np.max(rates)+xbuffer]
    if ylims is None:
        ybuffer = 0.5
        ylims = [np.min(e_forms)-ybuffer, np.max(e_forms)+ybuffer]

    ax_scatter.set_xlim(xlims)
    ax_scatter.set_ylim(ylims)
    ax_eform_hist.set_ylim(ylims)
    ax_rate_hist.set_xlim(xlims)

    num_bins = hist_kwargs.pop("num_bins", 100)
    e_form_bins = np.linspace(ylims[0], ylims[1], num_bins)
    rate_bins = np.linspace(xlims[0], xlims[1], num_bins)

    hist_kwargs_ref = {
        "rate":{"bins":rate_bins},
        "e_form":{"bins":e_form_bins}
    }

    #plot the ref_distributions
    ref_hist_summary = plot_scatter_and_hists(
        datadicts=ref_datadicts,
        ax_scatter=ax_scatter,
        ax_rate_hist=ax_rate_hist,
        ax_eform_hist=ax_eform_hist,
        use_log=use_log,
        color="C0",
        alpha=alpha,
        hist_kwargs=hist_kwargs_ref,
        scatter_kwargs=scatter_kwargs
    )

    if datadicts_add is not None:
        if datadicts_add_colors is None:
            datadicts_add_colors = [f"C{i+2}" for i in range(len(datadicts_add))]
        for i, datadict_add, color in zip(range(len(datadicts_add)), datadicts_add, datadicts_add_colors):
            plot_scatter_and_hists(
                datadicts=datadict_add,
                ax_scatter=ax_scatter,
                ax_rate_hist=ax_rate_hist,
                ax_eform_hist=ax_eform_hist,
                use_log=use_log,
                color=color,
                alpha=alpha,
                zorder_scatter=1+i,
                zorder_hist=-1*(i+1),
                scatter_kwargs=scatter_kwargs,
                hist_kwargs=hist_kwargs_ref
            )

    if percentile is not None:
        rates, e_forms = get_rates_and_eforms_from_datadicts(
            datadicts=ref_datadicts,
            use_log=use_log
        )
        rate_percentile = np.percentile(rates, percentile)
        e_form_percentile = np.percentile(e_forms, 100.0-percentile)
        ax_scatter.vlines(rate_percentile, ylims[0], ylims[1], colors="k", linewidths=1, linestyles="--")
        ax_scatter.hlines(e_form_percentile, xlims[0], xlims[1], colors="k", linewidths=1, linestyles="--")
        if highligt_percentile_region:
            ax_scatter.fill_between(
                x=[rate_percentile,  xlims[1]],
                y1=[ylims[0], ylims[0]],
                y2=[e_form_percentile, e_form_percentile],
                color="grey",
                edgecolor='black',
                linewidth=0.5,
                alpha=0.25,
                hatch="x",
                zorder=-100
            )

def embed_elements_as_onehot(
        elements:list,
        element_mapping_dict:dict
    ):
    element_onehot = np.array([
        embed_idx_to_onehot(
            idx=element_mapping_dict[element],
            elem_population=len(element_mapping_dict)
        ) 
        for element in elements
    ])
    return element_onehot

def embed_idx_to_onehot(
        idx:int,
        elem_population:int
    ):
    result = np.zeros(elem_population)
    result[idx]+=1
    return result

def entropy_of_coverage(
        frac_coverage,
        log_reg:float=1e-9
    ):
    return -np.sum(frac_coverage*np.log(frac_coverage+log_reg))/np.log(16.0)


def swap_reference_energy(
        e_form_old:float,
        symbols:list,
        energies_ref_old:dict,
        energies_ref_new:dict,
    ):
    e_form = e_form_old
    stoichiometries = dict(Counter(symbols))
    old_offset = sum(stoichiometries[ee] * energies_ref_old[ee] for ee in stoichiometries)
    e_form += old_offset
    new_offset = sum(stoichiometries[ee] * energies_ref_new[ee] for ee in stoichiometries)
    e_form -= new_offset
    return e_form

def swap_reference_energies(
        datadicts:list,
        energies_ref_old:dict,
        energies_ref_new:dict,
    ):
    for datadict in datadicts:
        symbols = datadict["elements"]
        e_form = datadict["e_form"]
        e_form_new = swap_reference_energy(
            e_form_old=e_form,
            symbols=symbols,
            energies_ref_old=energies_ref_old,
            energies_ref_new=energies_ref_new
        )
        datadict["e_form"] = e_form_new


def entropy_of_coverage(
        frac_coverage,
        log_reg:float=1e-9,
        element_pool:list=None,
        normalize:bool=True
    ):
    c = 1.0
    if normalize and element_pool is not None:
        c = np.log(len(element_pool))
    return -np.sum(frac_coverage*np.log(frac_coverage+log_reg))/c