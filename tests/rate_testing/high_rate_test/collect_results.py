from ase.io import read
from gen_catalyst_design.db import Database, load_data_from_db
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    scenarios = [
        "active_site_encoding",
        "no_active_site_encoding"
    ]

    models = [
        "Absorbing",
        "Uniform"
    ]

    guidance_scales = [0.8, 1.2, 2.0]

    training_data = read("training_set.traj", index=":")
    rates_training = np.array([atoms.info["rate"] for atoms in training_data])
    rate_min = 1.0
    rate_interval = 0.5
    rate_max_training = np.max(rates_training)
    bins = np.arange(rate_min, rate_max_training, rate_interval)
    hist, edges_training = np.histogram(rates_training, bins=bins)
    hist_norm_training = hist/np.sum(hist)

    for scenario in scenarios:
        for model in models:
            fig, axs = plt.subplots(len(guidance_scales), 1, sharex=True, layout="constrained", figsize=(9,6))
            for guidance_scale, ax in zip(guidance_scales, axs):
                ax.stairs(hist_norm_training, edges=edges_training, fill=True, alpha=0.7, edgecolor="k", linewidth=1, color="C1", label="Training data")
                ax.tick_params(axis='x', direction='inout', length=8)
                ax.set_title(f"guidance scale: {guidance_scale}")
                ax.set_ylabel("Normalized distribution %")
            
                sample_db_file = f"g_{guidance_scale}_scale.db"
                pth_header = os.path.join(scenario, model, "samples")

                database = Database.establish_connection(
                    filename=sample_db_file,
                    miller_index="100",
                    pth_header=pth_header
                )

                datadicts = load_data_from_db(database=database)
                rates = np.array([datadict["rate"] for datadict in datadicts])
                bins = np.arange(0.0, np.ceil(np.max(rates)), rate_interval)
                hist, edges = np.histogram(rates, bins=bins)
                hist_norm = hist/np.sum(hist)
                ax.stairs(hist_norm, edges=edges, fill=True, alpha=0.7, edgecolor="k", linewidth=1, color="C0", label="generated samples")
                ax.legend()
            #ax.set_xticks(classes)
            plt.savefig(f"{scenario}_{model}.png")
            plt.close()

if __name__ == "__main__":
    main()