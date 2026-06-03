import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml


def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    # ---- Replace with your actual data ----

    with open("stochiometries.yaml", "r") as fileobj:
        tot_data = yaml.safe_load(fileobj)

    stoch_data = tot_data["stochiometries"]
    rate_bins = list(stoch_data.keys())
    entropies = []
    sites = list(range(21))
    rate_intervals = []
    for rate_bin in rate_bins:
        rate_intervals.append(stoch_data[rate_bin]["rate_division"])
        entropy_per_site = []
        for site in sites:
            entropy_per_site.append(stoch_data[rate_bin][f"site_{site}"]["entropy"])
        entropies.append(entropy_per_site)
    
    rate_intervals = np.array(rate_intervals)
    #print(rate_intervals)
    entropies = np.array(entropies)
  
    fig, ax = plt.subplots(figsize=(7,4))

    im = ax.imshow(
        entropies,
        aspect='auto',
        cmap='YlGn', #YlGn
        vmin=np.round(np.min(entropies),1)-0.2,        # use if normalized entropy
        vmax=1.0#np.round(np.max(entropies)         # use if normalized entropy
    )

    ax.set_xticks(np.arange(-0.5, entropies.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, entropies.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Axis labels
    ax.set_xlabel("Site")
    ax.set_ylabel(r"$\log_{10}$(rate)")
    #ax.set_title("Entropy of Element Distribution per Site")

    # Ticks
    ax.set_xticks(np.arange(len(sites)))
    ax.set_xticklabels(sites)
    print(np.log10(np.mean(rate_intervals,axis=1)))

    yticks = range(0,len(rate_bins), 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(np.mean(np.log10(rate_intervals),axis=1), 2)[::2])
    #ax.set_aspect("equal")
    ax.invert_yaxis()
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized entropy")

    plt.tight_layout()
    plt.savefig("entropy_100.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()