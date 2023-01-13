"""
ModelAngelo refine command.
If you already have a built model that you would like to refine using ModelAngelo
or if you have built poly-alanine chains that you would like HMM profiles for,
you can use this command.

You will need:
1) A PDB/mmCIF file with a built model, provide with --input-structure/--i/-i
2) A cryo-EM map .mrc file, provide with --volume-path/--v/-v
3) (Optional) An output directory for the results, provide with --output-dir/--o/-o
4) (Optional) If you would like HMM profile outputs, provide the flag --write-hmm-profiles/--w/-w
"""

import argparse
import json
import os
import shutil

import torch

from model_angelo.data.standardize_mrc import standardize_mrc
from model_angelo.gnn.inference import infer as gnn_infer
from model_angelo.utils.misc_utils import setup_logger, write_relion_job_exit_status, filter_useless_warnings, Args, \
    abort_if_relion_abort
from model_angelo.utils.torch_utils import download_and_install_model, get_device_name


def add_args(parser):
    main_args = parser.add_argument_group(
        "Main arguments",
        description="These are the only arguments a typical user will need."
    )
    main_args.add_argument(
        "--input-structure",
        "-i",
        "--i",
        help="Input structure with the built model.",
        type=str,
        required=True,
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
        "--write-hmm-profiles",
        "-w",
        "--w",
        action="store_true",
        help="Write HMM profiles for the chains."
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
        default="original",
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

        abort_if_relion_abort(parsed_args.output_dir)

        # Returns a list
        assert (
                len(standarized_mrc_path) > 0
        ), f"standardize_mrc did not get any inputs: {standardize_mrc_args.input_path}"
        standarized_mrc_path = standarized_mrc_path[0]

        # Run GNN inference --------------------------------------------------------------------------------------------

        print(f"------------------------ GNN model refinement ------------------------")

        current_output_dir = os.path.join(parsed_args.output_dir, "gnn_output_round_1")
        os.makedirs(current_output_dir, exist_ok=True)

        gnn_infer_args = Args(config["gnn_infer_args"])
        gnn_infer_args.map = standarized_mrc_path
        gnn_infer_args.fasta = None
        gnn_infer_args.rna_fasta = None
        gnn_infer_args.dna_fasta = None
        gnn_infer_args.struct = parsed_args.input_structure
        gnn_infer_args.output_dir = current_output_dir
        gnn_infer_args.model_dir = gnn_model_logdir
        gnn_infer_args.device = parsed_args.device
        gnn_infer_args.write_hmm_profiles = parsed_args.write_hmm_profiles
        gnn_infer_args.refine = True

        gnn_infer_args.aggressive_pruning = True

        logger.info(f"GNN model refinement round with args: {gnn_infer_args}")
        gnn_output = gnn_infer(gnn_infer_args)

        abort_if_relion_abort(parsed_args.output_dir)

        hmm_profiles_src = os.path.join(os.path.dirname(gnn_output), "hmm_profiles")

        name = os.path.basename(parsed_args.output_dir)
        file_dst = os.path.join(parsed_args.output_dir, f"{name}_refined.cif")
        hmm_profiles_dst = os.path.join(parsed_args.output_dir, "hmm_profiles")

        os.replace(gnn_output, file_dst)

        if parsed_args.write_hmm_profiles:
            shutil.rmtree(hmm_profiles_dst, ignore_errors=True)
            os.replace(hmm_profiles_src, hmm_profiles_dst)

        os.remove(standarized_mrc_path)

        print("-" * 70)
        print("ModelAngelo refine has been completed successfully!")
        print("-" * 70)
        print(f"You can find your output mmCIF file here: {file_dst}")
        print("-" * 70)

        if parsed_args.write_hmm_profiles:
            print(
                f"The HMM profiles are available in the directory: {hmm_profiles_dst}\n"
                f"They are named according to the chains found in {file_dst}\n"
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
