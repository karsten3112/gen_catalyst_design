from gen_catalyst_design.db import Database, load_data_from_db
from gen_catalyst_design.discrete_space_diffusion import DiffusionModel
from gen_catalyst_design.reaction_rates import ReactionMechanism
from gen_catalyst_design.utils import get_calculator, get_features_bulk_and_gas, reaction_rate_of_RDS_from_symbols
from ase.db import connect
from ase_ml_models.databases import get_atoms_list_from_db
import matplotlib.pyplot as plt
import torch
import os
import numpy as np


def main():
    miller_index = "100"
    pth_header = f"../../results/random_search/results/Rh_Cu_Au_Pd/miller_index_{miller_index}"
    run_ids = 1
    rate_min = 0.0
    num_classes = 30
    pred_index = 10
    random_seed = 42

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


    datadicts = []
    for runid in range(run_ids):
        filename = f"runID_{runid}_results.db"
        database = Database.establish_connection(filename=filename, miller_index=miller_index, pth_header=pth_header)
        datadicts += load_data_from_db(database=database)

    filtered_dicts = filter_data_dicts(data_dicts=datadicts, rate_min=rate_min)
    rates = [data["rate"] for data in filtered_dicts]

    class_ranges = np.linspace(rate_min, np.max(rates), num_classes)
    diff = class_ranges[1] - class_ranges[0]
    hist, bin_edges = np.histogram(rates, bins=class_ranges)
    norm = np.sum(hist*diff)
    fig, ax = plt.subplots()
    ax.stairs(hist/norm, bin_edges, fill=True, alpha=0.6, edgecolor="k", linewidth=1)
   
    diff_model = DiffusionModel.load_from_checkpoint(f"AbsorbingStateDiffusion/model_010/checkpoints/epoch=epoch=42-val=val_loss=0.0078.ckpt")
    diff_model = diff_model.to(device=torch.device("cpu"))#torch.zeros(size=(10,), dtype=torch.long)#

    db = connect("../../databases/templates/100/100_templates.db")
    template_atoms_list = get_atoms_list_from_db(db_ase=db)
    n_samples = 100
    conditionings = pred_index*torch.ones(size=(n_samples,), dtype=torch.long)
    result_samples = diff_model.sample(
        guidance_scale=0.85,
        n_samples=n_samples, 
        conditionings=conditionings, 
        template_atoms=template_atoms_list[0], 
        batch_size=50, 
        timesteps=None, 
        log_all_timesteps=False, 
        return_as_atoms_list=True
    )
    print(len(result_samples))
    score_dicts = reaction_rate_calculation(
        samples=result_samples,
        template_atoms_list=template_atoms_list,
        n_atoms_surf=len(template_atoms_list[0]),
        miller_index=miller_index,
        universal_pth_header="../../"
    )
    rates = [score_dict["rate"] for score_dict in score_dicts]
    hist, bin_edges = np.histogram(rates, bins=class_ranges)
    norm = np.sum(hist*diff)
    ax.stairs(hist/norm, bin_edges, fill=True, alpha=0.6, edgecolor="k", linewidth=1)
    plt.savefig("distribution.png")


def reaction_rate_calculation(
        samples:list,
        template_atoms_list:list,
        n_atoms_surf:int,
        miller_index:str,
        model:str="WWL-GPR",
        universal_pth_header:str="../",
    ):
    features_bulk, features_gas = get_features_bulk_and_gas(pth_header=os.path.join(universal_pth_header, "yaml_files/features"))
    #Get calculator of model type and training parameters
    calculator, train_kwargs = get_calculator(model=model, miller_index=miller_index)
    
    #Train calculator on database
    calculator.train_model_from_db(
        db_filename=f"atoms_adsorbates_{miller_index}_DFT.db", 
        features_bulk=features_bulk, 
        features_gas=features_gas, 
        db_pth_header=os.path.join(universal_pth_header,"databases/DFT_database"),
        train_kwargs=train_kwargs
    )
    
    #setup reaction mechanism for calculating rate of RDS
    reaction_mechanism = ReactionMechanism(
        calculator=calculator,
        mechanism_pth_header=os.path.join(universal_pth_header,"yaml_files/reaction_mechanism")
    )
    score_dicts = []
    for sample in samples:
        score_dict = reaction_rate_of_RDS_from_symbols(
            reaction_mechanism=reaction_mechanism,
            symbols=sample[0].get_chemical_symbols(),
            template_atoms_list=template_atoms_list,
            features_bulk=features_bulk,
            features_gas=features_gas,
            n_atoms_surf=n_atoms_surf
        )
        score_dicts.append(score_dict)
    return score_dicts


def filter_data_dicts(data_dicts:list, rate_min:float=None):
    if rate_min is None:
        return data_dicts
    else:
        filtered_datadicts = []
        for datadict in data_dicts:
            if datadict["rate"] < rate_min:
                pass
            else:
                filtered_datadicts.append(datadict)
        return filtered_datadicts


if __name__ == "__main__":
    main()