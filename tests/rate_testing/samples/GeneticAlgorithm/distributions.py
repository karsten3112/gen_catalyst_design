from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import sys
from gen_catalyst_design.db import Database, get_atoms_list_db, load_data_from_db
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def main():
    miller_index = "100"
    pth_header = f"../../../../results/GeneticAlgorithm/results/Rh_Cu_Au_Pd/miller_index_{miller_index}"
    run_ids = [0,1,2]
    datadicts = []
    for run_id in run_ids:
        filename = f"runID_{run_id}_results.db"
        database = Database.establish_connection(filename=filename, miller_index=miller_index, pth_header=pth_header)
        datadicts += load_data_from_db(database=database)

    rates_training = np.array([datadict["rate"] for datadict in datadicts])
    #datadicts = sorted(datadicts, key=lambda xx: xx["rate"], reverse=False)

    rate_max_training = np.max(rates_training)
    step = 2.5
    rate_min = 1.0
    rate_interval = 0.5
    classes = np.arange(1.0, np.ceil(rate_max_training)+step, step)
    bins = np.arange(rate_min, rate_max_training, rate_interval)
    hist, edges = np.histogram(rates_training, bins=bins)
    hist_norm_training = hist/np.sum(hist)

    samples_dir = "model_005/rate_evals"
    classes_for_plot = [7]
    y_lim = 0.3
    for cls in classes_for_plot:
        fig, axs = plt.subplots(5,1, figsize=(8,12), sharex=True, layout="constrained")
        for ax in axs:
            ax.set_ylim([0.0, y_lim])
            ax.bar(classes[cls]+2.5/2.0, 1.0, fill=False, hatch="xx", width=step,linewidth=1, label="Conditioned-class")
            ax.set_xticks(classes)
            ax.tick_params(axis='x', direction='inout', length=8)
            ax.stairs(hist_norm_training, edges=edges, fill=True, alpha=0.7, edgecolor="k", linewidth=1, color="C1", label="Training data")
        class_path = os.path.join(samples_dir, f"class_{cls}")
        guidance_scales = os.listdir(class_path)
        i = 0
        for ax, filename in zip(axs, guidance_scales):
            splitted_name = filename.split("_")
            ax.set_title(f"guidance scale: {splitted_name[1]}")
            rates = load_rates_from_file(
                filename=filename,
                pth_header=os.path.join(class_path)
            )
            bins = np.arange(0.0, rate_max_training, rate_interval)
            hist, edges = np.histogram(rates, bins=bins)
            hist_norm = hist/np.sum(hist)
            ax.stairs(hist_norm, edges=edges, fill=True, alpha=0.7, edgecolor="k", linewidth=1, color="C0", label="generated samples")
            ax.set_ylabel("Normalized distribution %")
            if i == 0:
                ax.legend()
            i+=1
        ax.set_xlabel("Rate [1/s]")
        plt.savefig("distribution.png")
        plt.close()

    
def load_rates_from_file(
    filename:str,
    pth_header:str,
    ):
    database = Database.establish_connection(
        filename=filename,
        miller_index="100",
        pth_header=pth_header
    )
    datadicts = load_data_from_db(database=database)
    rates = np.array([datadict["rate"] for datadict in datadicts])
    return rates


if __name__ == "__main__":
    main()