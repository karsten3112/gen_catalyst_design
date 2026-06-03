from gen_catalyst_design.utils import get_full_element_pool_no_saas, get_atom_color_dict, get_element_hatches
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml




def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    element_pool = get_full_element_pool_no_saas()
    element_colors = get_atom_color_dict(element_pool=element_pool)

    element_hatch = get_element_hatches(element_pool=element_pool)

    use_log = True
    show_top_k_coverages = 4
    miller_index = "100"

    with open("stochiometries.yaml", "r") as fileobj:
        tot_data = yaml.safe_load(fileobj)

    element_mapping_list= list(tot_data["element_mapping_dict"].keys())
    stoch_data = tot_data["stochiometries"]
    rate_bins = list(stoch_data.keys())
    rate_intervals = np.array([stoch_data[rate_bin]["rate_division"] for rate_bin in rate_bins])

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(len(rate_bins), 2, width_ratios=[1, 3.0], wspace=0.05, hspace=0.05)
    axs_left = [fig.add_subplot(gs[0, 0])]
    axs_left += [
        fig.add_subplot(gs[i, 0], sharex=axs_left[0])
        for i in range(1, len(rate_bins))
    ]

    axs_right = [fig.add_subplot(gs[0, 1], sharey=axs_left[0])]
    axs_right += [
        fig.add_subplot(gs[i+1, 1], sharex=axs_right[0], sharey=ax_left)
        for i, ax_left in enumerate(axs_left[:-1])
    ]

    for ax in axs_left[:-1]:
        ax.tick_params(labelbottom=False)

    axs_right[0].tick_params(labelleft=False)
    for ax in axs_right[:-1]:
        ax.tick_params(labelbottom=False, labelleft=False)
    


    sites_left = [0,1,2,3]
    sites_right = [len(sites_left)+i for i in range(21-len(sites_left))]
    
    for ax, sites in zip([axs_left[0], axs_right[0]], [sites_left, sites_right]):
        ax.set_xticks(ticks=sites, labels=sites)



    for i, axs, sites in zip(range(2), [axs_left, axs_right], [sites_left, sites_right]):
        for ax, rate_bin in zip(axs, rate_bins):
            for site in sites:
                if f"site_{site}" in stoch_data[rate_bin]:
                    coverages = np.array(stoch_data[rate_bin][f"site_{site}"]["site_coverage"])
                    #print(type(coverages))
                    indices = np.flip(np.argpartition(coverages, show_top_k_coverages)[-show_top_k_coverages:])
                    for j, index in enumerate(indices):
                        ax.bar(x=site, height=coverages[index]*100, color=element_colors[element_mapping_list[index]], edgecolor="k", hatch=element_hatch[element_mapping_list[index]])
                        if j == 0:
                            ax.text(
                                site,
                                coverages[index]*100 + 0.02,   # small vertical offset (adjust as needed)
                                f"{np.sum(coverages[indices])*100:.1f}\%",
                                ha='center',
                                va='bottom',
                                fontsize=10 if i == 0 else 9
                            )
            ax.set_xlim([np.min(sites)-0.5, np.max(sites)+0.5])
            if i == 0:
                ax.set_ylabel(f"Frequency of element \%")
                ax.set_xlabel("Active sites")
            else:
                ax.set_xlabel("Neighboring sites")


    legend_handles = [
        Patch(
            facecolor=element_colors[el], 
            label=el,
            edgecolor="k",
            hatch=element_hatch[el]
        )
        for el in element_pool
    ]

    fig.legend(
        handles=legend_handles,
        #title="Element",
        bbox_to_anchor=(0.2, 0.0),
        loc="upper left",
        ncol=6,# if row_plot else len(elements)
        #borderaxespad=0.0,
    )

    fig.suptitle(
        r"Top 80'th percentile, "+ r"rate span: "+rf"{np.round(np.log10(rate_intervals[0][0]),2)}-{np.round(np.log10(rate_intervals[0][1]),2)} "+ r"[$\log_{10}$(rate)]," + f" N samples: {stoch_data[rate_bin]['num_samples']}",
        y=0.95,
        bbox=dict(
            facecolor="white",
            edgecolor="grey",
            linewidth=1,
            alpha=0.8,
            boxstyle="round,pad=0.25"
        )
    )

    plt.savefig("top_20_ga.pdf", bbox_inches="tight")





if __name__ == "__main__":
    main()


