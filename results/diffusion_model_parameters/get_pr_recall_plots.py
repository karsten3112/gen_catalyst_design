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

    drop_prob = 0.1 #[0.1, 0.2, 0.5]
    pth_header = "hyperparam_search"
    models = [model_name for model_name in os.listdir(pth_header) if "model" in model_name]
    tau = 0
    add_scheduler_and_noiser_plots = False


    diff_model_params = get_diff_model_params(
        models=models,
        pth_header=pth_header
    )

    model_color_dict = get_diffmodel_colors(
        diffmodel_params_dict=diff_model_params
    )

    if add_scheduler_and_noiser_plots:
        fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(12,5), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(layout="constrained", figsize=(5,5))
        axs = [ax]
    
    g_scale_markers = {0.5:"o", 1.0:"^", 2.0:"*", 4.0:"s", 8.0:"D"}
    g_scale_colors = {
        0.5:"C0",
        1.0:"C1",
        2.0:"C2",
        4.0:"C3",
        8.0:"C4"
    }
    noiser_colors = {
        "AbsorbingStateNoiser":"C0",
        "UniformTransitionsNoiser":"C1"
    }
    scheduler_colors = {
        "ExponentialBetaScheduler0.05":"C0",
        "ExponentialBetaScheduler1.0":"C1",
        "CosineScheduler":"C2",
        "LinearBetaScheduler":"C3",
        "LinearAlphaScheduler":"C4"
    }
    print(model_color_dict)
    axs[0].set_ylabel("Precision \%")
    for i, ax in enumerate(axs):
        ax.set_xlabel("Coverage \%")
        ax.grid()
        for model in diff_model_params:
            drop_prob_model = diff_model_params[model]["drop_prob"]
            noiser = diff_model_params[model]["noiser_info"]["noiser_type"]
            scheduler = diff_model_params[model]["scheduler_info"]["scheduler_type"]
            if scheduler == "ExponentialBetaScheduler":
                scheduler+=str(diff_model_params[model]["scheduler_info"]["beta_max"])

            if drop_prob_model == drop_prob:
                with open(os.path.join(pth_header, model, "pr_recall.yaml")) as fileobj:
                    pr_recall_dict = yaml.safe_load(fileobj)

                with open(os.path.join(pth_header, model, "best_frac.yaml")) as fileobj:
                    best_frac_dict = yaml.safe_load(fileobj)
                
                precisions, recalls, g_scales, novelties, best_fracs = [], [], [], [], []
                for g_scale_label in pr_recall_dict:
                    if pr_recall_dict[g_scale_label] is not None:
                        for pr_recall_info in pr_recall_dict[g_scale_label]:
                            if pr_recall_info["tau"] == tau:
                                g_scales.append(float(g_scale_label.split("_")[-1]))
                                precisions.append(pr_recall_info["precision"])
                                recalls.append(pr_recall_info["recall"])
                                novelties.append(pr_recall_info["novelty"])
                    
                    if best_frac_dict[g_scale_label] is not None:
                        best_fracs.append(best_frac_dict[g_scale_label])
                
                print(model, round(np.mean(best_fracs)*1000, 3))#round(np.mean(precisions), 3), round(np.mean(recalls), 3))
                for j, g_scale in enumerate(g_scales):
                    plot_kwargs = dict(
                        alpha=0.9, 
                        markeredgecolor="k", 
                        marker=g_scale_markers[g_scale],
                        markersize=8
                    )
                    if i == 0:
                        color = model_color_dict[model]
                    elif i == 1:
                        color = noiser_colors[noiser]
                    elif i == 2:
                        color = scheduler_colors[scheduler]
                    else:
                        raise Exception("no color is defined for this plot")
                    ax.plot(recalls[j]*100, precisions[j]*100,color=color, **plot_kwargs)
    ax.set_title(rf"$\tau={tau}$ [Tokens]")
    plt.savefig(f"pr_coverage_tau_{tau}.pdf")
    #print(diff_model_params["model_0"].keys())



def get_diff_model_params(
        models:list,
        pth_header:str=None
    ):
    result_param_dict = {}
    for model in models:
        if pth_header is not None:
            model_dir = os.path.join(pth_header, model)
        else:
            model_dir = model
        param_filename = os.path.join(model_dir, "wandb", "latest-run", "files", "model_parameters.yaml")
        with open(param_filename, "r") as fileobj:
            parameters = yaml.safe_load(fileobj)
        result_param_dict[model] = parameters
    return result_param_dict

def get_diffmodel_colors(
        diffmodel_params_dict:dict
    ):
    color_dict = {}
    models = list(diffmodel_params_dict.keys())
    setting_combinations =  {}
    i=0
    for model in models:
        settings = diffmodel_params_dict[model]
        scheduler_info = settings["scheduler_info"]
        setting_comb = scheduler_info["scheduler_type"] + settings["noiser_info"]["noiser_type"]
        if "beta_max" in scheduler_info:
            setting_comb+=str(scheduler_info["beta_max"])
        if setting_comb not in setting_combinations:
            setting_combinations[setting_comb] = i
            i+=1
    color_dict = {}
    for model in models:
        settings = diffmodel_params_dict[model]
        scheduler_info = settings["scheduler_info"]
        setting_comb = scheduler_info["scheduler_type"] + settings["noiser_info"]["noiser_type"]
        if "beta_max" in scheduler_info:
            setting_comb+=str(scheduler_info["beta_max"])
        color_dict[model] = f"C{setting_combinations[setting_comb]}"
    return color_dict
    #color_list = [f"C{i}" for i in range(len(setting_combinations))]




            

if __name__ == "__main__":
    main()