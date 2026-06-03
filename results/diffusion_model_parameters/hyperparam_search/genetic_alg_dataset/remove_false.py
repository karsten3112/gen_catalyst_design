import os
import yaml
import sys
import shutil


def main():
    pth_header = "../../../genetic_alg_dataset"
    yaml_file_header = "wandb/latest-run/files"
    drop_prob = 0.1
    models = [model for model in os.listdir(pth_header) if "model" in model]
    noiser = "UniformTransitionsNoiser"
    i = 5
    for model in models:
        yaml_file = os.path.join(pth_header, model, yaml_file_header, "model_parameters.yaml")
        with open(yaml_file, "r") as fileobj:
            model_params = yaml.safe_load(fileobj)
        model_drop_prob = model_params["drop_prob"]
        model_noiser = model_params["noiser_info"]["noiser_type"]
        if model_drop_prob == drop_prob and noiser == model_noiser:
            if not os.path.exists(f"model_{i}"):
                os.makedirs(f"model_{i}")
            filenames = os.listdir(os.path.join(pth_header, model))
            for filename in filenames:
                src = os.path.join(pth_header, model, filename)
                dst = os.path.join(f"model_{i}")
                #if os.path.isdir(src):
                shutil.move(src, dst)
                #else:
                #    shutil.copy(src, dst)
            i+=1

if __name__ == "__main__":
    main()


