import matplotlib.pyplot as plt
import yaml


def main():
    stat_measures = [
        "mean",
        "median",
        "IQR",
        "max_val"
    ]

    fig, axs = plt.subplots(1, len(stat_measures), layout="constrained", figsize=(10,4))

    with open("summary_desc.yaml", mode="r") as fileobj: 
        hyperparameter_statistics = yaml.safe_load(fileobj)


    for stat_measure, ax in zip(stat_measures, axs):
        ax.set_ylabel(stat_measure)
        xs = []
        values = []
        errs = []
        for i, hyper_set in enumerate(hyperparameter_statistics):
            stat_dict = hyperparameter_statistics[hyper_set][stat_measure]
            value = stat_dict["mean"]
            err = stat_dict["err"]
            values.append(value)
            errs.append(err)
            xs.append(i)
        
        ax.errorbar(xs, values, errs, capsize=10.0, fmt="o-")

    plt.savefig("ill_log.png")
if __name__ == "__main__":
    main()