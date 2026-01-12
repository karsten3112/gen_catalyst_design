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
    beam_search(
        diff_model=diff_model,
        template_atoms=template_atoms,
        n_samples=100,
        n_branches=10,
        top_k_solutions=2
    )


def beam_search(diff_model:DiffusionModel, template_atoms, n_samples, top_k_solutions, n_branches, guidance_scale:float=0.0):
    
    n_elements = len(template_atoms)
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
    for i, timestep in enumerate(timesteps):
        guided_probs = get_guided_probs(diff_model=diff_model, sample_loader=sample_loader, timestep=timestep, guidance_scale=guidance_scale)
        if i == 0:
            guided_probs = guided_probs.reshape(n_samples, n_elements, len(diff_model.element_pool))
        else:
            guided_probs = guided_probs.reshape(n_samples, top_k_solutions, n_elements, len(diff_model.element_pool))
        branched_probs = guided_probs.repeat_interleave(n_branches, dim=0)
        branched_samples = diff_model.sample_onehot_vectors(probabilities=branched_probs)
        print(guided_probs.shape)
        print(branched_samples.shape)
        scores = (branched_samples*torch.log(branched_probs)).sum(dim=-1).sum(dim=-1).reshape(n_samples, n_branches)
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k_solutions, dim=-1)
        top_k_samples = torch.gather(
            branched_samples.reshape(n_samples, n_branches, n_elements, len(diff_model.element_pool)), dim=1,
            index=top_k_indices[:, :, None, None].expand(-1, -1, 21, 5)
        )
        #top_k_samples = branched_samples.reshape(n_samples, n_branches, n_elements, len(diff_model.element_pool))[top_k_indices]
        print(top_k_samples.shape)
        #ids = torch.ones(size=(100,10))
        #ids*guided_probs
        #branch_samples(guided_probs=guided_probs, n_branches=n_brances)
        exit()

def get_guided_probs(diff_model:DiffusionModel, sample_loader, timestep, guidance_scale) -> torch.tensor:
    guided_probs = torch.vstack([
        diff_model.get_reverse_transition_probabilities(
            batch = batch,
            time=torch.ones(size=(batch.batch_size,), dtype=torch.long)*timestep,
            guidance_scale=guidance_scale
            ) 
    for batch in sample_loader]
    )
    return guided_probs


def get_dataloader():
    pass


def calculate_scores():
    pass

def filter_top_k_samples():
    pass


if __name__ == "__main__":
    main()