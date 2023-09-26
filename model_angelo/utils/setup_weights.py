import argparse
import json
import os

from model_angelo.utils.torch_utils import (
    download_and_install_model,
    download_and_install_esm_model,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ModelAngelo weights installation utility")
    parser.add_argument("--bundle-name", type=str, default="original")
    
    args = parser.parse_args()
    
    print(
        "Please make sure you have set the environment variable TORCH_HOME \n"
        "to a suitable directory, visible to all relevant users!"
    )
    
    model_dst = download_and_install_model(args.bundle_name)
    
    with open(os.path.join(model_dst, "config.json"), "r") as f:
        config = json.load(f)
    
    gnn_infer_args = config["gnn_infer_args"]
    if "esm_model" in gnn_infer_args:
        download_and_install_esm_model(gnn_infer_args["esm_model"])
    
    print("Successful!")
    print(f"Path to model is {model_dst}")
