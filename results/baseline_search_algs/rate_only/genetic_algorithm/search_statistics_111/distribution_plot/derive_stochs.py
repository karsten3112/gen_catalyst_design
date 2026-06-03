from gen_catalyst_design.db import Database, load_datadicts_from_db_with_rate_selection, load_datadicts_from_db
from gen_catalyst_design.utils import get_full_element_pool_no_saas
from gen_catalyst_design.discrete_space_diffusion.Dataset import embed_elements_as_onehot
from ase_ml_models.yaml import write_to_yaml
import numpy as np
import yaml



def main():
    filename = "genetic_algorithm_no_rot_filter.db"
    element_pool = get_full_element_pool_no_saas()
    element_mapping_dict = {element:idx for idx, element in enumerate(element_pool)}
    db = Database.establish_connection(
        filename=filename,
        pth_header=".."
    )

    datadicts = load_datadicts_from_db(database=db, close_connection=False)
    rates = np.array([datadict["rate"] for datadict in datadicts])
    step = 0.04
    percentiles = [0.8,1.0]#[0.8,1.0]#np.linspace(0.8, 1.0, 4)
    #print(percentiles)
    interval_spacings = np.array([np.percentile(rates, percentile*100) for percentile in percentiles])
    num_bins = len(interval_spacings)
    statistics_dict = {}
    statistics_dict["element_mapping_dict"] = element_mapping_dict
    #statistics_dict["rate_divisions"] = {idx:[interval_spacings[idx], interval_spacings[idx+1]] for idx in range(num_bins-1)}
    summary_dict = {}
    for i in range(num_bins-1):
        print(i, [interval_spacings[i], interval_spacings[i+1]])
        summary_dict[f"rate_bin_idx_{i}"] = {}
        summary_dict[f"rate_bin_idx_{i}"]["rate_division"] = [interval_spacings[i], interval_spacings[i+1]]
        result_dicts = load_datadicts_from_db_with_rate_selection(
            database=db,
            rate_min=interval_spacings[i],
            rate_max=interval_spacings[i+1],
            close_connection=False
        )
        summary_dict[f"rate_bin_idx_{i}"]["num_samples"] = len(result_dicts)
        if len(result_dicts) == 0:
            pass
        else:
            onehots = np.array([
                embed_elements_as_onehot(
                    elements=datadict["elements"],
                    element_mapping_dict=element_mapping_dict
                ) for datadict in result_dicts
            ])
            num_samples = len(onehots)
            summed_onehots = np.sum(onehots, axis=0)
            fractional_coverages = summed_onehots/num_samples
            
            for j, site_coverage in enumerate(fractional_coverages):
                site_entropy = entropy_of_coverage(frac_coverage=site_coverage)
                summary_dict[f"rate_bin_idx_{i}"][f"site_{j}"] = {"site_coverage":site_coverage, "entropy":site_entropy}
                #summary_dict[f"rate_bin_idx_{i}"][f"site_{j}"] = site_coverage

    statistics_dict["stochiometries"] = summary_dict
    write_to_yaml(
        filename="stochiometries.yaml",
        data=statistics_dict
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

if __name__ == "__main__":
    main()