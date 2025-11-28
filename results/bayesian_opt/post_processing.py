import sys
sys.path.insert(0, "../../")
from gen_catalyst_toolkit.utilities.databases import connect, write_data_to_db, load_data_from_db, clear_database
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    miller_index = "100"
    db_pth_header = f"results/miller_index_{miller_index}"
    databases = [connect(filename=file, pth_header=db_pth_header) for file in os.listdir(db_pth_header)]
    rates = []
    for database in databases:
        data_dicts = load_data_from_db(database=database)
        for data in data_dicts:
            rates.append(data["scores"]["TOF"])
    rates = np.array(rates)
    bins = np.arange(np.min(rates), np.max(rates), 1e-1)
    hist, bin_edges = np.histogram(rates, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = (bin_centers[1] - bin_centers[0])
    fig, ax = plt.subplots()
    ax.bar(x=bin_centers, height=hist, width=bin_width)
    ax.stairs(hist, bin_edges, color="C1")
    plt.savefig("hej.png")


    



if __name__ == "__main__":
    main()