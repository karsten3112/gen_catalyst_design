from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import sys
from gen_catalyst_design.db import Database, get_atoms_list_db, load_data_from_db
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ase.io import read
import matplotlib.pyplot as plt
import os


def main():
    element_pool = ["O"] + ["Au","Cu","Pd","Rh","Ni","Ga"] #Oxygen added to represent aborbing state
    dim_reduction_method = "TSNE"

    classes = [0, 1, 2]

    dataset_atoms = read("../dataset.traj", index=":")
    class_divisions = np.array([atoms.info["class"] for atoms in dataset_atoms])
    
    #class_indices_dict = {cls:np.argwhere(class_divisions == cls).squeeze() for cls in classes}
    training_set_embedded = get_onehots_from_atoms_list(atoms_list=dataset_atoms, element_pool=element_pool)
    
    if dim_reduction_method == "TSNE":
        dim_reducer = TSNE(
            n_components=2, 
            learning_rate='auto', 
            init='random',
            #random_state=100
        )
    else:
        dim_reducer = PCA(n_components=2)
        #dim_reducer.explained_variance_ratio_

    #training_set_dim_reduced = dim_reducer.fit_transform(training_set_embedded)
    #for cls in class_indices_dict:
    #    indices = class_indices_dict[cls]
    #    ax.scatter(training_set_dim_reduced[indices][:,0], training_set_dim_reduced[indices][:,1], c=f"C{cls}")

    model = "model_004"
    guidance_scales = [0.8]
    
    denoised_trajes = {}

    training_data_size = len(training_set_embedded)

    for guidance_scale in guidance_scales:
        all_embedded_data = training_set_embedded
        final_class_divisions = class_divisions
        fig, ax = plt.subplots()
        classes = os.listdir(model)
        for cls in classes:
            samples_dir = os.path.join(model, cls, f"g_{guidance_scale}_scale")
            denoise_traj_files = os.listdir(samples_dir)
            cls_num = int(cls.split("_")[-1])
            for denoise_file in denoise_traj_files:
                traj_pth = os.path.join(samples_dir, denoise_file)
                denoise_traj = read(filename=traj_pth, index="::10")
                embedded_traj = get_onehots_from_atoms_list(atoms_list=denoise_traj, element_pool=element_pool)
                all_embedded_data = np.vstack([all_embedded_data, embedded_traj])
                final_class_divisions = np.hstack([final_class_divisions, cls_num*np.ones(shape=(len(denoise_traj,)), dtype=int)])
        
        dim_reduced_data = dim_reducer.fit_transform(all_embedded_data)
        if dim_reduction_method == "PCA":
            print(dim_reducer.explained_variance_ratio_)
        training_reduced = dim_reduced_data[:training_data_size-1]
        training_class_indices = final_class_divisions[:training_data_size-1]
        training_class_indices_dict = {cls:np.argwhere(training_class_indices == cls).squeeze() for cls in [0,1,2]}
        for cls in training_class_indices_dict:
            indices = training_class_indices_dict[cls]
            ax.scatter(training_reduced[indices][:,0], training_reduced[indices][:,1], c=f"C{cls}", alpha=0.3)
       
        trajes_reduced =  dim_reduced_data[training_data_size:].reshape(len(classes), 10, 100, 2)
        cls_index, traj_nums, _, _, = trajes_reduced.shape
        for cls in range(cls_index):
            for traj_num in range(traj_nums):
                ax.plot(trajes_reduced[cls,traj_num,:,0], trajes_reduced[cls,traj_num,:,1], "o-", c=f"C{cls}", alpha=0.3)
    

    plt.savefig(f"{dim_reduction_method}.png")
    plt.close()


def element_to_onehot(element, mapping_dict):
    onehot = np.zeros(len(mapping_dict))
    onehot[mapping_dict[element]] +=1
    return onehot

def get_onehots_from_datadicts(datadicts, element_pool):
    mapping_dict = {element:i for i, element in enumerate(element_pool)}
    result_onehots = []
    for datadict in datadicts:
        elements = datadict["elements"]
        onehots = np.hstack([element_to_onehot(element=element, mapping_dict=mapping_dict) for element in elements])
        result_onehots.append(onehots)
    return np.vstack(result_onehots)

def get_onehots_from_atoms_list(atoms_list, element_pool):
    mapping_dict = {element:i for i, element in enumerate(element_pool)}
    result_onehots = []
    for atoms in atoms_list:
        symbols = atoms.get_chemical_symbols()
        onehots = np.hstack([element_to_onehot(element=element, mapping_dict=mapping_dict) for element in symbols])
        result_onehots.append(onehots)
    return np.vstack(result_onehots)


if __name__ == "__main__":
    main()