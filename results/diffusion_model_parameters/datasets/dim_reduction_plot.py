from gen_catalyst_design.utils import get_full_element_pool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.atoms import Atoms
from ase.io import read
import numpy as np

def main():
    #mpl.rcParams['text.usetex'] = True
    #mpl.rcParams['font.family'] = 'serif'
    #mpl.rcParams["font.size"] = 14
    #mpl.rcParams["ytick.labelsize"] = 12
    #mpl.rcParams["xtick.labelsize"] = 12


    dataset = "genetic_algorithm_2000.traj"
    dim_reduce_method = "pca"
    use_log = True
    cmap = plt.cm.magma

    element_pool = get_full_element_pool()
    idx_embed_dict = {element:i for i, element in enumerate(element_pool)}
    atoms_list = read(dataset, ":")
    onehot_embeddings = np.vstack(
        [embed_atoms_concat_onehot(atoms=atoms, idx_embed_dict=idx_embed_dict) for atoms in atoms_list]
    )
    rates = np.array([atoms.info["rate"] for atoms in atoms_list])
    if use_log:
        rates = np.log10(rates)

    rate_min, rate_max = np.min(rates), np.max(rates)
    print(rate_min, rate_max)
    fig, ax = plt.subplots()
    if dim_reduce_method == "pca":
        dim_reducer = PCA(n_components=2)
        proj_x = dim_reducer.fit_transform(onehot_embeddings)
    elif dim_reduce_method == "tsne":
        X = np.asarray(onehot_embeddings, dtype=np.float64, order="C")
        dim_reducer = TSNE(
            n_components=2,
            random_state=42,
            learning_rate=200.0,
            init="pca",
            perplexity=30,
            method="exact",
            verbose=2,
        )
        proj_x = dim_reducer.fit_transform(X)
    else:
        raise Exception(f"no method of type {dim_reduce_method} is implemented")


    scatter = ax.scatter(proj_x[:,0], proj_x[:,1], c=rates, cmap=cmap, vmin=rate_min, vmax=rate_max, alpha=0.6)
    plt.colorbar(scatter, label="Rate [1/s]")
    plt.savefig(f"{dim_reduce_method}.png")
    #print(onehot_embeddings.shape)



def embed_elem_onehot(
        element:str,
        idx_embed_dict:dict,
    ):
    onehot = np.zeros(shape=(len(idx_embed_dict),))
    onehot[idx_embed_dict[element]]+=1
    return onehot*1.0


def embed_atoms_concat_onehot(
        atoms:Atoms,
        idx_embed_dict:dict
    ): 
    return np.hstack([embed_elem_onehot(element=element, idx_embed_dict=idx_embed_dict) for element in atoms.get_chemical_symbols()])



if __name__ == "__main__":
    main()