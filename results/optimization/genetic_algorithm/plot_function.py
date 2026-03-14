import matplotlib.pyplot as plt
from gen_catalyst_design.db import Database, load_datadicts_from_db
import numpy as np

def main():
    fig, ax = plt.subplots(figsize=(12,5))
    db = Database.establish_connection(
        "test_opt.db"
    )
    #plot_distribution_boxplot(ax=ax,database=db, num_distributions=11, logy=False)
    plot_distribution_violin(ax=ax, database=db, num_distributions=100, logy=False)
    #plot_distribution_curve(ax=ax,database=db)
    plt.savefig("test.png")



def plot_distribution_curve(
    ax,
    database:Database,
    num_distributions:int=11,
    stepsize=0.5,
    ):
    #ax.set_ylim([1000,2000])
    #ax.set_yscale("log")
    datadicts = load_datadicts_from_db(database=database)
    all_rates = get_rates_from_datadicts(datadicts)
    bins = np.arange(np.min(all_rates), np.max(all_rates) + stepsize, stepsize)

    for ii in range(num_distributions):
        data_distribution = [
            datadict for datadict in datadicts
            if datadict["gen_iter"] == ii
        ]

        rates = get_rates_from_datadicts(data_distribution)
        if len(rates) == 0:
            continue

        hist, bin_edges = np.histogram(rates, bins=bins)
        if hist.sum() == 0:
            continue

        hist_norm = hist / hist.sum()*5.0
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        ax.barh(
            y=bin_centers,
            width=hist_norm,
            height=bin_widths,
            left=ii,              # displace each distribution along x by ii
            align="center",
            edgecolor="black",    # lines around each bin
            linewidth=1.0,
            alpha=0.8,
        )



def plot_distribution_boxplot(
    ax,
    database: Database,
    num_distributions: int = 5,
    logy: bool = True,
):
    datadicts = load_datadicts_from_db(database=database)

    for ii in range(num_distributions):
        data_distribution = [
            datadict for datadict in datadicts
            if datadict["gen_iter"] == ii
        ]

        rates = np.asarray(get_rates_from_datadicts(data_distribution))
        if len(rates) == 0:
            continue

        if logy:
            rates = rates[rates > 0]
            if len(rates) == 0:
                continue

        ax.boxplot(
            rates,
            positions=[ii],
            widths=0.6,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightgray", edgecolor="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(
                marker="o",
                markersize=4,
                markerfacecolor="black",
                markeredgecolor="black",
            ),
        )

    if logy:
        ax.set_yscale("log")

    #ax.set_xlim(-0.5, num_distributions - 0.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("rate")

def plot_distribution_violin(
    ax,
    database: Database,
    num_distributions: int = 5,
    logy: bool = True,
):
    datadicts = load_datadicts_from_db(database=database)

    positions = []
    datasets = []
    j = 0
    for ii in range(num_distributions):
        if ii % 10 == 0:
            data_distribution = [
                datadict for datadict in datadicts
                if datadict["gen_iter"] == ii
            ]

            rates = np.asarray(get_rates_from_datadicts(data_distribution))
            print(np.mean(rates))
            if len(rates) == 0:
                continue

            if logy:
                rates = rates[rates > 0]
                if len(rates) == 0:
                    continue

            positions.append(j)
            datasets.append(rates)
            j+=1

    if len(datasets) == 0:
        return

    parts = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )

    for body in parts["bodies"]:
        body.set_facecolor("lightgray")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")

    if logy:
        ax.set_yscale("log")

    #ax.set_xlim(-0.5, num_distributions - 0.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("rate")


def get_rates_from_datadicts(
        datadicts:list
    ):
    rates = np.array([datadict["rate"] for datadict in datadicts])
    return rates


if __name__ == "__main__":
    main()