from gen_catalyst_design.discrete_space_diffusion import DiffusionModel, AbsorbingStateNoiser, CosineScheduler, MPNNLogitPredictor
from gen_catalyst_design.discrete_space_diffusion.Dataset import get_dataloaders_from_atoms_list
from ase.io import read, write
import torch
import torch.nn.functional as F



def main():
    torch.manual_seed(42)
    element_pool = ["(X)"] + ['Ni', 'Cu', 'Rh', 'Ir', 'Pd', 'Pt', 'Au', 'Ag']
    atoms_list = read("../../../results/reconstruction_check/chgnet_result_all_fcc.traj", index="0:100")

    train_loader, val_loader = get_dataloaders_from_atoms_list(
        atoms_list=atoms_list,
        element_pool=element_pool,
        batch_size=4,
        condition_keys=["e_form", "rate"]
    )

    noiser = AbsorbingStateNoiser(
        element_pool=element_pool
    )

    scheduler = CosineScheduler()

    #noiser.pre_compute_accum_q_matrices(
    #    scheduler=scheduler
    #)

    #exit()

    logit_predictor = MPNNLogitPredictor(
        num_elements=len(element_pool),
        conditioning_dim=64
    )

    diff_model = DiffusionModel(
        element_pool=element_pool,
        scheduler=scheduler,
        noiser=noiser,
        logit_predictor=logit_predictor
    )


    for batch in train_loader:
        batch_copy = batch.clone()
        time = torch.ones(batch.batch_size, dtype=torch.int)*500

        probs = noiser.get_accum_transition_probabilities(
            x0_batch=batch.x*1.0,
            time_batch=time[batch.batch]
        )

        print(probs)

        exit()

        noiser.noise_batch_x0_xt(
            batch=batch_copy,
            time_batch=time[batch.batch]
        )

        denoise_logits = diff_model.get_guided_logits(
            batch=batch_copy,
            time=time,
            guidance_scale_dict={}
        )

        denoise_probs = logit_predictor.get_probs_from_logits(
            logits=denoise_logits
        )

        x0s = [
            F.one_hot(torch.tensor(i), num_classes=len(element_pool)) * \
            torch.ones(size=(len(batch.x), 1)) 
            for i, elem in enumerate(element_pool) #if elem != "(X)"
        ]

        if noiser.absorbing_state_index is not None:
            indices = torch.argmax(batch_copy.x, dim=-1)
            mask = indices != noiser.absorbing_state_index
            onehots = F.one_hot(indices, num_classes=len(element_pool))
            denoise_probs[mask]*= onehots[mask]
            denoise_probs[mask]/=denoise_probs[mask].sum(dim=-1, keepdim=True)


        q_revs_tot = torch.stack([noiser.get_reverse_transition_probabilities(
            x0_batch=x0*1.0,
            x_t_batch=batch_copy.x*1.0, 
            time_batch=time[batch.batch], 
            scheduler=scheduler
        ) for x0 in x0s
        ])
    
        weights = denoise_probs.T.unsqueeze(-1)
        reverse_probs = (weights * q_revs_tot).sum(dim=0)


def get_valid_x0s(
        xt_batch
    ):
    torch.argmax(xt_batch, dim=-1)



if __name__ == "__main__":
    main()