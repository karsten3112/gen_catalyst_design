
def get_features_bulk_and_gas(
        bulk_filename:str="features_bulk.yaml", 
        gas_filename:str="features_gas.yaml", 
        pth_header:str=None
        ) -> tuple:
        """
        Get features for bulk and gas phase.
        """
        if pth_header is not None:
            bulk_filename = os.path.join(pth_header, bulk_filename)
            gas_filename = os.path.join(pth_header, gas_filename)
        # Read features from yaml files.
        with open(bulk_filename, "r") as fileobj:
            features_bulk = yaml.safe_load(fileobj)

        with open(gas_filename, "r") as fileobj:
            features_gas = yaml.safe_load(fileobj)
        # Preprocess features.
        features_bulk = preprocess_features(features_dict=features_bulk)
        features_gas = preprocess_features(features_dict=features_gas)
        # Return parameters.
        return features_bulk, features_gas

def get_calculator(model, miller_index):
    from gen_catalyst_design.calculators import GCNNCalculator, GraphCalculator
    if model == "WWL-GPR":
        calculator = GraphCalculator(
             miller_index=miller_index,
             kernel="GPR"
        )
        train_kwargs = {}
    elif model == "GCNN":
        calculator = GCNNCalculator(
            miller_index=miller_index
        )
        network_hyperparams = {"hidden_dim":128,
            "n_conv_layers": 4,
            "n_lin_layers": 2,
            "conv_type": "ARMAConv",
            "dropout":0.0,
            "activation": torch.nn.functional.relu,
            #"use_batch_norm":False,
            #"use_residual":False
        }
        train_kwargs = {
                    "early_stop":True,
                    "num_epochs":500,
                    "lr": 1e-4,
                    "batch_size": 16,
                    "target": "E_form",
                    "val_split": 0.1,
                    "early_stopping_patience": 100,
                    "early_stopping_delta": 1e-4,
                    "weight_decay":0.0,
                    "hyperparams":network_hyperparams
        }
    else:
         raise Exception(f"calculator of type: {model} has not been implemented yet")
    return calculator, train_kwargs


def reaction_rate_of_RDS_from_symbols(
    reaction_mechanism:ReactionMechanism,
    symbols: list,
    template_atoms_list: list,
    features_bulk: dict,
    features_gas: dict,
    n_atoms_surf: int,
):
    """
    Get reaction rate of the RDS from the surface symbols.
    """
    # Update elements and features of atoms for predictions.
    update_atoms_list(
        atoms_list=template_atoms_list,
        features_bulk=features_bulk,
        features_gas=features_gas,
        symbols=symbols,
        n_atoms_surf=n_atoms_surf,
    )
    # Predict formation energies with a calculator.
    # Calculate reaction rate from reaction mechanism.
    score_dict = reaction_mechanism(atoms_list=template_atoms_list)
    return score_dict