from gen_catalyst_design.db import Database, load_datadicts_from_db_with_rate_selection
from ase_ml_models.yaml import write_to_yaml
import numpy as np
import yaml
import os



def main():
    miller_index = "100"
    surface_type = "cluster"
    top_k_solutions = 10
    use_log = True

    with open("../distribution_plot/stochiometries.yaml", "r") as fileobj:
        tot_stoch_data = yaml.safe_load(fileobj)

    rate_intervals = tot_stoch_data["stochiometries"]["rate_bin_idx_0"]["rate_division"]

    db = Database.establish_connection(
        filename="genetic_algorithm_no_rot_filter.db",
        pth_header=".."
    )

    datadicts = load_datadicts_from_db_with_rate_selection(
        database=db,
        rate_max=rate_intervals[1],
        rate_min=rate_intervals[0]
    )

    unique_site_configurations = collect_unique_active_site_confs(
        datadicts=datadicts,
        miller_index=miller_index,
        surface_type=surface_type
    )

    top_k_abundances = get_most_abundant_active_site_configurations(
        unique_active_site_confs=unique_site_configurations,
        top_k_solutions=top_k_solutions,
        tot_amount_of_samples=len(datadicts),
        use_log=use_log
    )

    top_k_rates = get_highest_rate_active_site_configurations(
        unique_active_site_confs=unique_site_configurations,
        top_k_solutions=top_k_solutions,
        tot_amount_of_samples=len(datadicts),
        use_log=use_log
    )

    write_to_yaml(filename="top_k_abundances.yaml", data=top_k_abundances)
    write_to_yaml(filename="top_k_rates.yaml", data=top_k_rates)

    print(top_k_abundances)
    print(top_k_rates)

   # print(unique_site_configurations.keys())


def collect_unique_active_site_confs(
        datadicts:list,
        miller_index:str="100",
        surface_type:str="cluster"
    ):
    if surface_type == "cluster":
        if miller_index == "100":
            active_site_indices = [0,1,2,3]
        elif miller_index == "111":
            active_site_indices = []
        else:
            raise Exception(f"surface type of {surface_type} with miller-index {miller_index} is not implemented")
    elif surface_type == "surface":
        raise Exception(f"not implemented yet")
    
    filtered_dicts = {}
    for datadict in datadicts:
        elements = datadict["elements"]
        active_site_conf = "_".join([elements[idx] for idx in active_site_indices])
        if active_site_conf in filtered_dicts:
            filtered_dicts[active_site_conf].append(datadict["rate"])
        else:
            filtered_dicts[active_site_conf] = [datadict["rate"]]
    return filtered_dicts


def get_most_abundant_active_site_configurations(
        unique_active_site_confs:dict,
        top_k_solutions:int=10,
        tot_amount_of_samples:int=None,
        use_log:bool=True    
    ):
    active_site_len_dicts = [{"active_conf":active_conf, "n":len(unique_active_site_confs[active_conf])} for active_conf in unique_active_site_confs]
    sorted_lens = sorted(active_site_len_dicts, key=lambda x: x["n"], reverse=True)
    top_k_configurations = sorted_lens[:top_k_solutions]
    for top_conf_dict in top_k_configurations:
        rates = np.array(unique_active_site_confs[top_conf_dict["active_conf"]])
        if use_log:
            rates = np.log10(rates)

        top_conf_dict["mean_rate"] = np.mean(rates)
        top_conf_dict["std_rate"] = np.std(rates, ddof=1)
        top_conf_dict["max_rate"] = np.max(rates)
        if tot_amount_of_samples is not None:
            top_conf_dict["frac"] = top_conf_dict["n"]/tot_amount_of_samples
        
    return top_k_configurations

def get_highest_rate_active_site_configurations(
        unique_active_site_confs:dict,
        top_k_solutions:int=10,
        use_log:bool=True,
        tot_amount_of_samples:int=None
    ):
    active_site_max_dicts = []
    for active_conf in unique_active_site_confs:
        rates = np.array(unique_active_site_confs[active_conf])
        if use_log:
            rates = np.log10(rates)

        max_rate = np.max(rates)
        n = len(rates)
        mean = np.mean(rates)
        std = np.std(rates, ddof=1)
        store_dict = {"active_conf":active_conf, "max_rate":max_rate, "mean_rate":mean, "std":std}
        if tot_amount_of_samples is not None:
            store_dict["frac"] =n/tot_amount_of_samples
        store_dict["n"] = n
        active_site_max_dicts.append(store_dict)
    
    sorted_max_dicts = sorted(active_site_max_dicts, key=lambda x: x["max_rate"], reverse=True)
    top_k_rates = sorted_max_dicts[:top_k_solutions]
    return top_k_rates
    


if __name__ == "__main__":
    main()