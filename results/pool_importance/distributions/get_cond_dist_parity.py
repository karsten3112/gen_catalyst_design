from gen_catalyst_design.db import Database, load_datadicts_from_db
from gen_catalyst_design.post_processing import plot_rate_histogram
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from ase.io import read
import matplotlib as mpl
import numpy as np
import os





def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    use_log = False
    model = "model_no_att_pool_no_log"
    temp = 1.0
    alpha = 0.8

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_ylabel(r"Normalized counts \%")
    ax_left.set_xlabel(r"rate [1/s]")
    
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.set_xlabel(r"rate-conditioning [1/s]")
    ax_right.set_ylabel(r"rate-generated [1/s]")

    fig.suptitle(
        "Pure GNN",
        fontsize=16,
        y=0.93
    )

    train_atoms_list = read(
        filename=os.path.join("..", "no_duplicates.traj"),
        index=":"
    )
    train_datadicts = [{"elements":atoms.get_chemical_symbols(), "rate":atoms.info["rate"]}  for atoms in train_atoms_list]

    rate_conditions = get_rate_conditions(
        atoms_list=train_atoms_list,
        use_log=use_log,
        percentiles=[0.25, 0.5, 0.75, 0.95]
    )

    s = 1.0
    ax_right.plot(
        [np.min(rate_conditions)-s, np.max(rate_conditions)+s],
        [np.min(rate_conditions)-s, np.max(rate_conditions)+s],
        c="k",
        lw=1,
        ls="--"
    )


    bins = np.arange(0.0, 30.0, 0.5)
    plot_rate_histogram(
        ax=ax_left,
        datadicts=train_datadicts,
        bins=bins,
        apply_log=use_log,
        plot_kwargs={"alpha":alpha}
    )

    sample_dir = os.path.join("..", "trained_models", model, f"temp_{temp}_samples")
    spacing = 5.0
    for i, rate_condition in enumerate(rate_conditions):
        db = Database.establish_connection(
            filename=f"condition_{i+1}.db",
            pth_header=sample_dir
        )

        datadicts = load_datadicts_from_db(database=db)

        plot_rate_histogram(
            ax=ax_left,
            datadicts=datadicts,
            bins=bins,
            apply_log=use_log,
            bottom=spacing*(i+1),
            plot_kwargs={"zorder":-i, "color":f"C{i+1}", "alpha":alpha}
        )
        ax_left.annotate(
            "",
            xy=(rate_condition, 0),                  # arrow tip
            xytext=(rate_condition, spacing*(i+1)),  # arrow start
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                linestyle="--",
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3"
            )
        )
        ax_left.scatter(
            rate_condition, spacing*(i+1), zorder=10, c="k", s=10
        )
       
        plot_boxplot(
            ax=ax_right,
            datadicts=datadicts,
            placement=rate_condition,
            use_log=use_log,
            alpha=alpha,
            color=f"C{i+1}",
            plot_kwargs={"showfliers":False, "widths":1.5}
        )


    legend_elements = [
        Patch(facecolor="C0", edgecolor="black",
          linewidth=1.0,
          alpha=alpha,
          label=f"Training distribution"),
        Patch(facecolor="C1", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=r"$R_{25\%}$"),
        Patch(facecolor="C2", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=r"$R_{50\%}$"),
        Patch(facecolor="C3", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=r"$R_{75\%}$"),
        Patch(facecolor="C4", edgecolor="black",
          linewidth=1.0,
        alpha=alpha,
          label=r"$R_{95\%}$"),
        ]
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=12,
        frameon=True
    )


    plt.savefig(f"{model}.pdf", bbox_inches="tight")


def get_rate_conditions(
        atoms_list:list,
        use_log:bool=True,
        percentiles:np.array=None,
        percentile_kwargs:dict={}
    ):
    rates = np.array([atoms.info["rate"] for atoms in atoms_list])
    if use_log:
        rates = np.log10(rates)

    if percentiles is None:
        percentiles = np.linspace(
            percentile_kwargs.pop("lower_percent_lim",0.8), 
            percentile_kwargs.pop("upper_percent_lim",0.95), 
            percentile_kwargs.pop("num_conds", 5)
        )
    conditions = np.array([np.round(np.percentile(rates, percentile*100), 2) for percentile in percentiles])
    return conditions


def plot_boxplot(
        ax: object,
        datadicts: list,
        placement: float,
        use_log: bool = True,
        color:str="C0",
        alpha:float=0.8,
        plot_kwargs: dict = {}
    ):
    median_color = plot_kwargs.pop("median_color", "black")
    rates = np.array([datadict["rate"] for datadict in datadicts])
    if use_log:
        rates = np.log10(rates)

    box_plot_dict = ax.boxplot(
        rates,
        orientation="vertical",
        positions=[placement],
        patch_artist=True,
        manage_ticks=False,
        **plot_kwargs
    )

    # Set box color + alpha
    for box in box_plot_dict['boxes']:
        box.set_facecolor(color)
        box.set_alpha(alpha)

    # Set median color
    for median in box_plot_dict['medians']:
        median.set_color(median_color)


if __name__ == "__main__":
    main()