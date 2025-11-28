import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import sys
sys.path.insert(0, "../../../")



def main():
    plot_kwargs = {"ls":"-",
                   "marker":"o",
                   "markeredgecolor":"k",
                   "markersize":6.0,
                   }

    parameters = {#"ntasks-per-node":[1,2,3,4,5,6],
                     "mem-per-cpu":["3G"]#1G", "2G", "3G", "4G", "5G"],
                  #   "nodes":[1,2,3,4,5,6],
                  #   "partition":["q20","q24", "q28", "q36", "q40", "q48", "qgpu"]
                     }
    
    pth_header = "results"
    filename = "runID_42_timing_data.yaml"

    for param in parameters:
        fig, ax = plt.subplots()
        ax.set_xlabel("Samples [N]")
        ax.set_ylabel("seconds [s]")
        #ax.grid()
        #ax.set_title(param)
        for value in parameters[param]:
            filename_load = os.path.join(pth_header, f"timings_{param}",str(value), filename)
            with open(filename_load, "r") as fileobj:
                data_dict = yaml.safe_load(fileobj)
            log_times = np.array(data_dict["Logging_times_per_batch"])
            eval_times = np.array(data_dict["eval_times_per_batch"])
            training_time = data_dict["training_time"]
            time_per_batch = log_times + eval_times
            time_spent_arr = np.zeros(shape=time_per_batch.shape)
            time_spent = training_time
            for i, time in enumerate(time_per_batch):
                time_spent+=time
                time_spent_arr[i]=time_spent
            x_steps = np.arange(50, data_dict["batch_size"]*data_dict["batches"]+50, 50)
            eval_time_per_sample = np.mean(eval_times)/data_dict["batch_size"]/data_dict["batches"]
            ax.plot(x_steps, time_spent_arr, label=f"Bayesian-optimization", **plot_kwargs)
        
        ax.legend()
        plt.savefig(f"{param}.svg")




def plot_statistics(ax, parameter, x_values, data_dict, plot_kwargs:dict={}):
    ax.set_ylabel("[s] per sample.")
    plot_dict = {"rate-calculation":"eval_time", "logging":"logging_time"}
    ax.set_title(parameter)
    if parameter == "mem-per-cpu":
        x_values = [int(x_value[0]) for x_value in x_values]
    for label in plot_dict:
        data_plot = [get_time_per_sample(time=data, n_batches=data_dict["batches"][0], n_samples_per_batch=data_dict["batch_size"][0]) for data in data_dict[plot_dict[label]]]
        ax.plot(x_values, data_plot, label=label, **plot_kwargs)
    ax.legend()
    ax.grid()


def get_time_per_sample(time, n_batches, n_samples_per_batch):
    return time/n_batches/n_samples_per_batch


def get_sample_relation(training_times, eval_times_per_sample, log_times_per_sample):
    training_time = np.mean(training_times)
    eval_time_per_sample = np.mean(eval_times_per_sample)
    log_time_per_sample = np.mean(log_times_per_sample)
    print(f"relation for time-spent per sample eval t(n_samples): {training_time.round(3)}+({eval_time_per_sample.round(3)}+{log_time_per_sample.round(3)})*n_samples [s]" )


if __name__ == "__main__":
    main()