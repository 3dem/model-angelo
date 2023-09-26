import argparse
import json
import os

from model_angelo.utils.torch_utils import (
    download_and_install_model,
    download_and_install_esm_model,
)

def add_args(parser):
    """
    Need to remove model_bundle_path as a positional argument. It should not be required.
    It should normally reside in ~/.cache/model_angelo/bundle or something.
    """
    main_args = parser.add_argument_group(
        "Main arguments",
        description="These are the only arguments a typical user will need.",
    )
    main_args.add_argument("--bundle-name", type=str, default="original")
    return parser

def main(parsed_args):
    print(
        "Please make sure you have set the environment variable TORCH_HOME \n"
        "to a suitable directory, visible to all relevant users!"
    )
    
    model_dst = download_and_install_model(parsed_args.bundle_name)
    
    with open(os.path.join(model_dst, "config.json"), "r") as f:
        config = json.load(f)
    
    gnn_infer_args = config["gnn_infer_args"]
    if "esm_model" in gnn_infer_args:
        download_and_install_esm_model(gnn_infer_args["esm_model"])
    
    print("Successful!")
    print(f"Path to model is {model_dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__,
    )
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)
    
    
