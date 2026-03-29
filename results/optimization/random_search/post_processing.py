from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.post_processing import get_ecdf_sf, get_top_k_solutions, get_accummax_curve
from scipy.stats import iqr
import matplotlib.pyplot as plt
import numpy as np



def main():
    use_log = True
    results_dir = "results"
    rnd_seeds = list(range(10))

    survival_funcs = []
    top_k_results = []

    summary_dicts = []

    for rnd_seed in rnd_seeds:
        db = Database.establish_connection(
            filename=f"rnd_{rnd_seed}_seed.db",
            pth_header=results_dir
        )
        datadicts = load_datadicts_from_db(database=db)
        rate_distribution = np.array([datadict["rate"] for datadict in datadicts])
        if use_log:
            rate_distribution = np.log(rate_distribution)

        summary_dict = get_summary_dict(
            distribution=rate_distribution
        )
        summary_dicts.append(summary_dict)


def get_summary_dict(
        distribution:np.array,
        top_k:int=100
    ):
    summary_dict = {}

    ecdf, sf = get_ecdf_sf(
        distribution=distribution
    )

    top_k_solutions, indices = get_top_k_solutions(
        distribution=distribution,
        top_k=top_k
    )

def get_dist_summary(
        distribution:np.array,
        include_sf:bool=True,
        include_accum_max:bool=True   
    ):
    if include_sf:
        ecdf, sf = get_ecdf_sf(
            distribution=distribution
        )
    else:
        ecdf, sf = None, None

    if include_accum_max:
        accum_max = get_accummax_curve(
            distribution=distribution
        )
    else:
        accum_max = None
    
    median = np.median(distribution)
    mean = np.mean(distribution)
    IQR = iqr(distribution)

    return {"mean":mean, "median":median, "IQR":IQR, "sf":sf, "accum_max":accum_max}






        


if __name__ == "__main__":
    main()