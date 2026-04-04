from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.post_processing import get_tot_summary_dict, get_survival_func, get_accum_max_curve
from ase_ml_models.yaml import write_to_yaml
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os



def main():
    fig, axs = plt.subplots(1,2)
    use_log = True
    rnd_seeds = list(range(10))
    stat_measures = {
        "max_val":None,
        "unique_freq":None,
        "median":"whole_distribution",
        "IQR":"whole_distribution",
        "mean":"whole_distribution"
    }
    
    with open("hyperparams.yaml", mode="r") as fileobj:
        hyperparam_dicts = yaml.safe_load(fileobj)

    set_statistics_dict = {}

    for hyperparam_set in hyperparam_dicts:
        random_seeds = hyperparam_dicts[hyperparam_set]["-rnd_seeds"].split(",")
        summary_list = []
        for rnd_seed in random_seeds:
            db = Database.establish_connection(
                filename=f"rnd_seed_{rnd_seed}_samples.db",
                pth_header=hyperparam_set
            )
            datadicts = load_datadicts_from_db(database=db)
            rate_distribution = np.array([datadict["rate"] for datadict in datadicts])
            elements_list = [datadict["elements"] for datadict in datadicts]
        
            summary_dict = get_tot_summary_dict(
                distribution=rate_distribution,
                elements_list=elements_list,
                use_log=use_log
            )
            summary_list.append(summary_dict)

        mean_aggr_dict = {}
        for stat_measure in stat_measures:
            dist_belong = stat_measures[stat_measure]
            if dist_belong is None:
                values = np.array([summary[stat_measure] for summary in summary_list])
            else:
                values = np.array([summary[dist_belong][stat_measure] for summary in summary_list])
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            mean_aggr_dict[stat_measure] = {"mean":mean, "err":std/np.sqrt(len(values)),"std":std}

        set_statistics_dict[hyperparam_set] = mean_aggr_dict
    
    write_to_yaml(filename="summary_desc.yaml", data=set_statistics_dict)



if __name__ == "__main__":
    main()