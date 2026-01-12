from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from ase_ml_models.databases import get_atoms_list_from_db
from gen_catalyst_design.discrete_space_diffusion.Dataset import get_graph_from_atoms, GraphDataset
from ase.db import connect
from ase.io import read, write
import random
import torch
import os


def main():
    random_seed = 42
    n_samples = 100
    miller_index = "100"
    ckpt_file_type = "last"
    model = "model_005"

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    db = connect(f"../../../databases/templates/{miller_index}/{miller_index}_templates.db")
    template_atoms = get_atoms_list_from_db(db_ase=db)[0]

    ckpt_dir = os.path.join(model, "checkpoints")
    files = os.listdir(ckpt_dir)
    for file in files:
        if ckpt_file_type in file:
            ckpt_file = file

    diff_model = DiffusionModel.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file))
    diff_model = diff_model.to(device=torch.device("cpu"))

    samples = constrained_sampling(
        diff_model=diff_model,
        n_samples=n_samples,
        template_atoms=template_atoms,
        guidance_scale=0.0,
        active_sites=[0,1,2,3]
    )
    write("top_k_samples.traj", samples)


def constrained_sampling(
        diff_model:DiffusionModel,
        n_samples,
        template_atoms,
        guidance_scale,
        active_sites
    ):
    n_elements = len(template_atoms)
    timesteps = torch.arange(diff_model.scheduler.t_init, diff_model.scheduler.t_final+1, 1).flip(dims=(-1,))
    noised_atoms = diff_model.noiser.sample_atoms_from_stationary(
        n_samples=n_samples, 
        template_atoms=template_atoms
    )
    for atoms in noised_atoms:
        atoms[active_sites].symbols = ["Cu", "Cu", "Cu", "Cu"]
    sample_dataset = diff_model.denoiser.get_sample_dataset_from_atoms_list(
        atoms_list=noised_atoms,
        condition_key=None
    )
    sample_loader = diff_model.denoiser.get_sample_loader(
        dataset=sample_dataset,
        batch_size=n_samples,
        shuffle=False
    )
    element_index_count = {2:3, 4:1}
    samples = []
    for batch in sample_loader:
        for timestep in timesteps:
            guided_probs = diff_model.get_reverse_transition_probabilities(
                batch = batch,
                time=torch.ones(size=(batch.batch_size,), dtype=torch.long)*timestep,
                guidance_scale=guidance_scale
            )
            xs_denoised = diff_model.sample_onehot_vectors(probabilities=guided_probs)
            reshaped_xs = xs_denoised.reshape(n_samples, n_elements, len(diff_model.element_pool))
            apply_constraint()
            exit()

def apply_constraint(reshaped_xs, reshaped_probs, element_index_count, active_sites):
    active_site_count = reshaped_xs[:,active_sites].sum(dim=1)
    resample_probs = reshaped_probs*1.0
    for index in element_index_count:
        constraint_indices = active_site_count[:,index] > element_index_count[index]
        resample_probs[constraint_indices,active_sites,index]*=0.0
    return resample_probs


def top_k_sampling(
        diff_model:DiffusionModel, 
        n_samples, 
        template_atoms,
        top_k_solutions,
        guidance_scale
    ):
    timesteps = torch.arange(diff_model.scheduler.t_init, diff_model.scheduler.t_final+1, 1).flip(dims=(-1,))
    noised_atoms = diff_model.noiser.sample_atoms_from_stationary(
        n_samples=n_samples, 
        template_atoms=template_atoms
    )
    sample_dataset = diff_model.denoiser.get_sample_dataset_from_atoms_list(
        atoms_list=noised_atoms,
        condition_key=None
    )
    sample_loader = diff_model.denoiser.get_sample_loader(
        dataset=sample_dataset,
        batch_size=n_samples,
        shuffle=False
    )
    samples = []
    for batch in sample_loader:
        for timestep in timesteps:
            guided_probs = diff_model.get_reverse_transition_probabilities(
                batch = batch,
                time=torch.ones(size=(batch.batch_size,), dtype=torch.long)*timestep,
                guidance_scale=guidance_scale
            )
            xs_denoised = diff_model.sample_onehot_vectors(probabilities=probs_normed)
            top_k_scores, top_k_indices = torch.topk(guided_probs, k=top_k_solutions, dim=-1)
            probs = torch.zeros_like(guided_probs)
            probs.scatter_add_(dim=1, index=top_k_indices, src=top_k_scores)
            if timestep == diff_model.scheduler.t_init and diff_model.denoiser.absorbing_state:
                probs[:,diff_model.denoiser.absorbing_state_index] = 0.0
            probs_normed = probs/probs.sum(dim=-1, keepdim=True)
            xs_denoised = diff_model.sample_onehot_vectors(probabilities=probs_normed)
            batch.x = xs_denoised
        for i in range(batch.batch_size):
            data = batch.get_example(i)
            sample = data.to_atoms(diff_model.element_pool)
            samples.append(sample)
    return samples


if __name__ == "__main__":
    main()