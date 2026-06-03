import matplotlib.pyplot as plt
import numpy as np
import yaml



def main():
    with open("model_params.yaml", "r") as fileobj:
        sampling_summaries = yaml.safe_load(fileobj)
    fig, ax = plt.subplots()
    models = list(sampling_summaries.keys())
    for condition in ["condition_4"]:
        for model in models:
            g_scales = list(sampling_summaries[model].keys())
            mae_list = []
            g_scale_list = []
            for g_scale in g_scales:
                g_scale_val = float(g_scale.split("_")[-1])
                summary = sampling_summaries[model][g_scale][condition]
                g_scale_list.append(g_scale_val)
                mae_list.append(summary["mae"])
            ax.plot(g_scale_list, mae_list,"o-", label=model)
    plt.savefig("mae_plot.png")

    print(models)

if __name__ == "__main__":
    main()