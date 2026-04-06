import matplotlib.pyplot as plt
import yaml


def main():
    use_log = True
    stat_measures = {
        "mean":"Mean",
        "median":"Median",
        "IQR":"IQR",
        "max_val":"Maxmimum value"
    }


    for stat_measure in stat_measures:
        if use_log:
            stat_measures[stat_measure] += " [$\log($rate$)$]"
        else:
            stat_measures[stat_measure] += " [rate 1/s]"

    fig, axs = plt.subplots(1, len(stat_measures), layout="constrained", figsize=(12,4))

    with open("summary_desc.yaml", mode="r") as fileobj: 
        hyperparameter_statistics = yaml.safe_load(fileobj)


    for stat_measure, ax in zip(stat_measures, axs):
        ax.set_ylabel(stat_measures[stat_measure])
        xs = []
        values = []
        errs = []
        x_labels = []
        for i, hyper_set in enumerate(hyperparameter_statistics):
            if hyper_set != "default":
                splitted_label = hyper_set.split("_")
                x_labels.append(f"{splitted_label[0]}:" + f" {int(splitted_label[1])+1}")
            else:
                x_labels.append(f"{hyper_set}")
            
            stat_dict = hyperparameter_statistics[hyper_set][stat_measure]
            value = stat_dict["mean"]
            err = stat_dict["err"]
            values.append(value)
            errs.append(err)
            xs.append(i)

        ax.set_xticks(xs, x_labels, rotation=45)
        ax.errorbar(xs, values, errs, capsize=10.0, fmt="o-", markeredgecolor="k")

    plt.savefig("ill_log.png")
if __name__ == "__main__":
    main()