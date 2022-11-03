"""
Welcome to ModelAngelo! Have a look around...
This is the build_no_seq command, so you need:
1) A cryo-EM map .mrc file, passed to --volume-path/--v/-v

You can also provide a mask for the volume using --mask-path/--m/-m
You can also input a custom config file with --config-path/--c/-c

The output of this command will be less accurate than the
"build" command, however this command will also output HMM profiles
that can be used to search for potential sequences in your map
using HHBlits/HMMER.
"""

import argparse
import json
import os
import shutil
import sys

import torch

from model_angelo.c_alpha.inference import infer as c_alpha_infer
from model_angelo.data.standardize_mrc import standardize_mrc
from model_angelo.gnn.inference_no_seq import infer as gnn_no_seq_infer
from model_angelo.utils.misc_utils import filter_useless_warnings, setup_logger, Args, write_relion_job_exit_status
from model_angelo.utils.torch_utils import download_and_install_model, get_device_name


def add_args(parser):
    """
    Need to remove model_bundle_path as a positional argument. It should not be required.
    It should normally reside in ~/.cache/model_angelo/bundle or something.
    """
    main_args = parser.add_argument_group(
        "Main arguments",
        description="These are the only arguments a typical user will need."
    )
    main_args.add_argument(
        "--volume-path",
        "-v",
        "--v",
        help="input volume",
        type=str,
        required=True
    )
    main_args.add_argument(
        "--output-dir",
        "-o",
        "--o",
        help="path to output directory",
        type=str,
        default="output",
    )

    additional_args = parser.add_argument_group(
        "Additional arguments",
        description="These are sometimes useful."
    )
    additional_args.add_argument(
        "--mask-path", "-m", "--m", help="path to mask file", type=str, default=None
    )
    additional_args.add_argument(
        "--device",
        "-d",
        "--d",
        help="compute device, pick one of {cpu, gpu_number}. "
             "Default set to find an available GPU.",
        type=str,
        default=None,
    )

    advanced_args = parser.add_argument_group(
        "Advanced arguments",
        description="These should *not* be changed unless the user is aware of what they do."
    )
    advanced_args.add_argument(
        "--config-path", "-c", "--c", help="config file", type=str, default=None
    )
    advanced_args.add_argument(
        "--model-bundle-name",
        type=str,
        default="original_no_seq",
        help="Inference model bundle name",
    )
    advanced_args.add_argument(
        "--model-bundle-path",
        type=str,
        default=None,
        help="Inference model bundle path. If this is set, --model-bundle-name is not used."
    )

    # Below are RELION arguments, make sure to always add help=argparse.SUPPRESS

    parser.add_argument(
        "--pipeline-control",
        "--pipeline_control",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


def main(parsed_args):
    logger = setup_logger(os.path.join(parsed_args.output_dir, "model_angelo.log"))
    with logger.catch(
            message="Error in ModelAngelo",
            onerror=lambda _: write_relion_job_exit_status(
                parsed_args.output_dir,
                "FAILURE",
                pipeline_control=parsed_args.pipeline_control,
            )
    ):

        filter_useless_warnings()

        parsed_args.device = get_device_name(parsed_args.device)
        if parsed_args.model_bundle_path is None:
            model_bundle_path = download_and_install_model(parsed_args.model_bundle_name)
        else:
            model_bundle_path = parsed_args.model_bundle_path

        if parsed_args.config_path is None:
            config_path = os.path.join(model_bundle_path, "config.json")
        else:
            config_path = parsed_args.config_path
        with open(config_path, "r") as f:
            config = json.load(f)

        c_alpha_model_logdir = os.path.join(model_bundle_path, "c_alpha")
        gnn_model_logdir = os.path.join(model_bundle_path, "gnn")

        # Handle case with trailing /
        parsed_args.output_dir = os.path.normpath(parsed_args.output_dir)
        os.makedirs(parsed_args.output_dir, exist_ok=True)

        torch.no_grad()

        print("---------------------------- ModelAngelo -----------------------------")
        print("By Kiarash Jamali, Scheres Group, MRC Laboratory of Molecular Biology")

        logger.info(f"ModelAngelo with args: {vars(parsed_args)}")

        # Standarize input volume --------------------------------------------------------------------------------------

        standardize_mrc_args = Args(config["standardize_mrc_args"])
        standardize_mrc_args.input_path = (
            parsed_args.volume_path
        )  # The input file(s) to be standardized
        standardize_mrc_args.output_path = (
            parsed_args.output_dir
        )  # The output file(s) to be standardized

        logger.info(f"Input volume preprocessing with args: {standardize_mrc_args}")
        standarized_mrc_path = standardize_mrc(standardize_mrc_args)

        # Returns a list
        assert (
            len(standarized_mrc_path) > 0
        ), f"standardize_mrc did not get any inputs: {standardize_mrc_args.input_path}"
        standarized_mrc_path = standarized_mrc_path[0]

        # Run C-alpha inference ----------------------------------------------------------------------------------------
        print("--------------------- Initial C-alpha prediction ---------------------")

        ca_infer_args = Args(config["ca_infer_args"])
        ca_infer_args.log_dir = c_alpha_model_logdir
        ca_infer_args.model_checkpoint = "chkpt.torch"
        ca_infer_args.map_path = standarized_mrc_path
        ca_infer_args.output_path = os.path.join(parsed_args.output_dir, "see_alpha_output")
        ca_infer_args.mask_path = parsed_args.mask_path
        ca_infer_args.device = parsed_args.device
        ca_infer_args.auto_mask = (not ca_infer_args.dont_mask_input) and (
            parsed_args.mask_path is None
        )  # Use
        # automatically generated mask if no mask given

        logger.info(f"Initial C-alpha prediction with args: {ca_infer_args}")
        ca_cif_path = c_alpha_infer(ca_infer_args)

        # Run GNN inference --------------------------------------------------------------------------------------------

        current_ca_cif_path = ca_cif_path
        total_gnn_rounds = config["gnn_infer_args"]["num_rounds"]
        for i in range(total_gnn_rounds):
            print(f"------------------ GNN model refinement, round {i + 1} / {total_gnn_rounds} ------------------")

            current_output_dir = os.path.join(
                parsed_args.output_dir, f"gnn_output_round_{i + 1}"
            )
            os.makedirs(current_output_dir, exist_ok=True)

            gnn_infer_args = Args(config["gnn_infer_args"])
            gnn_infer_args.map = standarized_mrc_path
            gnn_infer_args.struct = current_ca_cif_path
            gnn_infer_args.output_dir = current_output_dir
            gnn_infer_args.model_dir = gnn_model_logdir
            gnn_infer_args.device = parsed_args.device

            if i == total_gnn_rounds - 1:
                gnn_infer_args.aggressive_pruning = True

            logger.info(f"GNN model refinement round {i + 1} with args: {gnn_infer_args}")
            gnn_output = gnn_no_seq_infer(gnn_infer_args)

            current_ca_cif_path = os.path.join(
                current_output_dir, "output.cif"
            )

        raw_file_src = gnn_output
        hmm_profiles_src = os.path.join(os.path.dirname(gnn_output), "hmm_profiles")

        name = os.path.basename(parsed_args.output_dir)
        raw_file_dst = os.path.join(parsed_args.output_dir, f"{name}.cif")
        hmm_profiles_dst = os.path.join(parsed_args.output_dir, "hmm_profiles")

        os.replace(raw_file_src, raw_file_dst)

        shutil.rmtree(hmm_profiles_dst, ignore_errors=True)
        os.replace(hmm_profiles_src, hmm_profiles_dst)

        os.remove(standarized_mrc_path)

        print("-" * 70)
        print("ModelAngelo build_no_seq has been completed successfully!")
        print("-" * 70)
        print(f"You can find your output mmCIF file here: {raw_file_dst}")
        print("-" * 70)
        print(
            f"The HMM profiles are available in the directory: {hmm_profiles_dst}\n"
            f"They are named according to the chains found in {raw_file_dst}\n"
            f"For example, chain A's profile is in {os.path.join(hmm_profiles_dst, 'A.hmm')}"
        )
        print(
            f"You can use model_angelo hmm_search to search these HMM profiles against a database"
        )
        print(
            f"Example command: \n"
            f"model_angelo hmm_search -i {parsed_args.output_dir} -f PATH_TO_DB.fasta "
            f"-o {os.path.join(parsed_args.output_dir, 'hmm_search_output')}"
        )
        print("-" * 70)
        print("Enjoy!")

        write_relion_job_exit_status(
            parsed_args.output_dir,
            "SUCCESS",
            pipeline_control=parsed_args.pipeline_control,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)
