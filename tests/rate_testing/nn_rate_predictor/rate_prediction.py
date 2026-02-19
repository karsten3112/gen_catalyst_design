from ase.io import read
from gen_catalyst_design.discrete_space_diffusion.guidance import ReactionRateModule
from gen_catalyst_design.discrete_space_diffusion.Dataset import get_dataloaders_from_atoms_list
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    model_type = "GNN"
    model = "3mpl_no_dup"
    ckpt_file_type = "epoch"
    add_active_site_connectivity = True
    element_pool = ["(X)"] + ["Rh", "Cu", "Au", "Pd"]

    ckpt_dir = os.path.join(model, "checkpoints")
    files = os.listdir(ckpt_dir)
    for file in files:
        if ckpt_file_type in file:
            ckpt_file = file

    reac_rate_model = ReactionRateModule.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file))
    reac_rate_model = reac_rate_model.to(device=torch.device("cpu"))
    
    train_loader, test_loader = get_dataloaders_from_atoms_list(
        atoms_list=read("../high_rates_no_duplicates.traj", index=":"),
        element_pool=element_pool,
        condition_key="rate",
        add_active_site_connectivity=add_active_site_connectivity,
        do_initial_shuffling=True,
        batch_size=40
    )
    
    true_rate_list = []
    pred_rate_list = []
    for batch in test_loader:
        true_rates = batch.y
        pred_rates = reac_rate_model.rate_prediction(x_t=batch.x*1.0, batch=batch)
        #print(pred_rates)
        true_rate_list.append(true_rates)
        pred_rate_list.append(pred_rates)
    true_rate_tensor = torch.cat(true_rate_list)
    pred_rate_tensor = torch.cat(pred_rate_list)
    fig, ax = plt.subplots()
    ax.set_xlabel("True rate [1/s]")
    ax.set_ylabel("Predicted rate [1/s]")
    with torch.no_grad():
        true_rates = true_rate_tensor.numpy()
        min_rate, max_rate = np.min(true_rates), np.max(true_rates)
        extra_spacing = 2
        ax.set_xlim([min_rate-extra_spacing, max_rate+extra_spacing])
        ax.set_ylim([min_rate-extra_spacing, max_rate+extra_spacing])
        line = np.linspace(min_rate-extra_spacing, max_rate+extra_spacing, 100)
        ax.plot(line, line, c="k", lw=1, ls="--")
        rmse = root_mean_squared_error(true_rates, pred_rate_tensor.numpy())
        mae = mean_absolute_error(true_rates, pred_rate_tensor.numpy())
        ax.scatter(true_rates, pred_rate_tensor.numpy(), alpha=0.6)
        ax.set_title(f"RMSE:{round(rmse, 3)}[1/s]   MAE:{round(mae, 3)}[1/s]")
    plt.savefig(f"rate_preds_{model}.png")
    plt.close()



if __name__ == "__main__":
    main()