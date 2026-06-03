from gen_catalyst_design.db import Database, load_datadicts_from_db
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml
import os




def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12

    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("Genetic Algorithm")

    hyperparams_file = "hyperparams.yaml"
    use_log = True
    show_mean = False
    spacing = 2.0
    plot_kwargs = {
        "showfliers":False, 
        "widths":1.0
    }

    with open(file=hyperparams_file, mode="r") as fileobj:
        hyperparams_data = yaml.safe_load(fileobj)

    datadicts_tot = get_datadicts_from_hyperparam_settings(
        hyperparams_data=hyperparams_data
    )

    if use_log:
        ax.set_ylabel(r"$\log_{10}$(rate)")
    else:
        ax.set_ylabel(r"rate [1/s]")

    x_ticks = [-1.0*spacing]
    x_ticklabels = ["Default"]
    for i, hyper_set in enumerate(datadicts_tot):
        if "_" in hyper_set:
            x_ticks.append(i*spacing)
            x_ticklabels.append(f"Set-{int(hyper_set.split('_')[-1])+1}")
    
    ax.set_xticks(
        ticks=x_ticks,
        labels=x_ticklabels,
        rotation=45,
        rotation_mode="anchor",
        ha="right",
        #va="top"
    )

    for i, hyper_set in enumerate(datadicts_tot):
        if hyper_set == "default":
            plot_boxplot(
                ax=ax,
                datadicts=datadicts_tot[hyper_set],
                placement=-1.0*spacing,
                use_log=use_log,
                plot_kwargs=plot_kwargs,
                color="C0",
                alpha=0.8
            )
        else:
            plot_boxplot(
                ax=ax,
                datadicts=datadicts_tot[hyper_set],
                placement=i*spacing,
                use_log=use_log,
                plot_kwargs=plot_kwargs,
                color="C1",
                alpha=0.8
            )
        if show_mean:
            mean = get_mean_of_datadicts(
                datadicts=datadicts_tot[hyper_set],
                use_log=use_log
            )
            ax.plot(i*spacing, mean, "o", c="k")

    plt.savefig("ga_hypers.pdf", bbox_inches="tight")


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


def get_mean_of_datadicts(
        datadicts:list,
        use_log:bool=True
    ):
    rates = np.array([datadict["rate"] for datadict in datadicts])
    if use_log:
        rates = np.log10(rates)
    return np.mean(rates)

def get_datadicts_from_hyperparam_settings(
        hyperparams_data:dict
    ):
    result_dict = {}
    for hyper_set in hyperparams_data:
        file_dir = hyperparams_data[hyper_set]["-dir"]
        rnd_seeds = [int(seed) for seed in hyperparams_data[hyper_set]["-rnd_seeds"].split(",")]
        datadicts = []
        for rnd_seed in rnd_seeds:
            db = Database.establish_connection(
                filename=f"rnd_seed_{rnd_seed}_samples.db",
                pth_header=file_dir
            )
            datadicts+=load_datadicts_from_db(database=db)
        result_dict[hyper_set] = datadicts
    return result_dict



if __name__ == "__main__":
    main()