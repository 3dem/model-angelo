"""
You are standing in an open field west of a glass building, with a sliding electric front door.
There is a small mailbox here.
> Open mailbox.
There is a letter.
> Read letter.
-------------------------------------------------------------------------
Welcome to ModelAngelo! Have a look around...
This is the build command, so you need:
1) A cryo-EM map .mrc file, passed to --volume-path/--v/-v
2) A FASTA file with all potential amino acid sequences found in the map,
   passed to --fasta-path/--f/-f
You can also provide a mask for the volume using --mask-path/--m/-m
You can also input a custom config file with --config-path/--c/-c
"""

import argparse
import json
import os
import shutil
import sys

import torch

from model_angelo.c_alpha.inference import infer as c_alpha_infer
from model_angelo.gnn.inference import infer as gnn_infer
from model_angelo.utils.fasta_utils import is_valid_fasta_ending, write_fasta_only_aa
from model_angelo.utils.misc_utils import (
    setup_logger,
    Args,
    is_relion_abort,
    write_relion_job_exit_status,
    abort_if_relion_abort, filter_useless_warnings, check_available_memory,
)
from model_angelo.utils.torch_utils import download_and_install_model, get_device_name


def add_args(parser):
    """
    Need to remove model_bundle_path as a positional argument. It should not be required.
    It should normally reside in ~/.cache/model_angelo/bundle or something.
    """
    main_args = parser.add_argument_group(
        "Main arguments",
        description="These are the only arguments a typical user will need.",
    )
    main_args.add_argument(
        "--volume-path", "-v", "--v", help="input volume", type=str, required=True
    )
    main_args.add_argument(
        "--protein-fasta",
        "--fasta-path",
        "--f",
        "--pf",
        "-f",
        "-pf",
        help="Protein input sequence FASTA file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rna-fasta",
        "--rf",
        "-rf",
        type=str,
        required=False,
        help="The path to the RNA sequence file",
    )
    parser.add_argument(
        "--dna-fasta",
        "--df",
        "-df",
        required=False,
        type=str,
        help="The path to the DNA sequence file",
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
        "Additional arguments", description="These are sometimes useful."
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
        description="These should *not* be changed unless the user is aware of what they do.",
    )
    advanced_args.add_argument(
        "--config-path", "-c", "--c", help="config file", type=str, default=None
    )
    advanced_args.add_argument(
        "--model-bundle-name",
        type=str,
        default="nucleotides",
        help="Inference model bundle name",
    )
    advanced_args.add_argument(
        "--model-bundle-path",
        type=str,
        default=None,
        help="Inference model bundle path. If this is set, --model-bundle-name is not used.",
    )
    advanced_args.add_argument(
        "--keep-intermediate-results",
        action="store_true",
        help="Keep intermediate results, ie see_alpha_output and gnn_round_x_output"
    )

    # Below are RELION arguments, make sure to always add help=argparse.SUPPRESS

    parser.add_argument(
        "--pipeline-control",
        "--pipeline_control",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )

    return parser



def main(parsed_args):
    filter_useless_warnings()
    logger = setup_logger(os.path.join(parsed_args.output_dir, "model_angelo.log"))
    with logger.catch(
        message="Error in ModelAngelo",
        onerror=lambda _: write_relion_job_exit_status(
            parsed_args.output_dir,
            "FAILURE",
            pipeline_control=parsed_args.pipeline_control,
        ),
    ):
        check_available_memory()
        if parsed_args.model_bundle_path is None:
            model_bundle_path = download_and_install_model(
                parsed_args.model_bundle_name
            )
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

        # Try to open FASTA       --------------------------------------------------------------------------------------
        from model_angelo.utils.fasta_utils import read_fasta
        new_protein_fasta_path = write_fasta_only_aa(parsed_args.protein_fasta)
        try:
            read_fasta(new_protein_fasta_path)
        except Exception as e:
            raise RuntimeError(
                f"File {parsed_args.protein_fasta} is not a valid FASTA file."
            ) from e

        # Run C-alpha inference ----------------------------------------------------------------------------------------
        print("--------------------- Initial C-alpha prediction ---------------------")

        ca_infer_args = Args(config["ca_infer_args"])
        ca_infer_args.log_dir = c_alpha_model_logdir
        ca_infer_args.model_checkpoint = "chkpt.torch"
        ca_infer_args.map_path = parsed_args.volume_path
        ca_infer_args.output_path = os.path.join(
            parsed_args.output_dir, "see_alpha_output"
        )
        ca_infer_args.mask_path = parsed_args.mask_path
        ca_infer_args.device = parsed_args.device
        ca_infer_args.auto_mask = (not ca_infer_args.dont_mask_input) and (
            parsed_args.mask_path is None
        )  # Use automatically generated mask if no mask given

        logger.info(f"Initial C-alpha prediction with args: {ca_infer_args}")
        ca_cif_path = c_alpha_infer(ca_infer_args)

        abort_if_relion_abort(parsed_args.output_dir)
        # Run GNN inference --------------------------------------------------------------------------------------------

        current_ca_cif_path = ca_cif_path
        total_gnn_rounds = config["gnn_infer_args"]["num_rounds"]
        for i in range(total_gnn_rounds):
            print(
                f"------------------ GNN model refinement, round {i + 1} / {total_gnn_rounds} ------------------"
            )

            current_output_dir = os.path.join(
                parsed_args.output_dir, f"gnn_output_round_{i + 1}"
            )
            os.makedirs(current_output_dir, exist_ok=True)

            gnn_infer_args = Args(config["gnn_infer_args"])
            gnn_infer_args.map = parsed_args.volume_path
            gnn_infer_args.protein_fasta = new_protein_fasta_path
            gnn_infer_args.rna_fasta = parsed_args.rna_fasta
            gnn_infer_args.dna_fasta = parsed_args.dna_fasta
            gnn_infer_args.struct = current_ca_cif_path
            gnn_infer_args.output_dir = current_output_dir
            gnn_infer_args.model_dir = gnn_model_logdir
            gnn_infer_args.device = parsed_args.device
            gnn_infer_args.write_hmm_profiles = False
            gnn_infer_args.refine = False

            if i == total_gnn_rounds - 1:
                if parsed_args.config_path is None:
                    gnn_infer_args.aggressive_pruning = True
                else:
                    gnn_infer_args.aggressive_pruning = config["gnn_infer_args"][
                        "aggressive_pruning"
                    ]
            else:
                gnn_infer_args.aggressive_pruning = False

            logger.info(
                f"GNN model refinement round {i + 1} with args: {gnn_infer_args}"
            )
            gnn_output = gnn_infer(gnn_infer_args)

            current_ca_cif_path = os.path.join(
                current_output_dir, "output.cif"
            )
            abort_if_relion_abort(parsed_args.output_dir)

        pruned_file_src = gnn_output.replace("output.cif", "output_fixed_aa_pruned.cif")
        raw_file_src = gnn_output.replace("output.cif", "output_fixed_aa.cif")
        
        name = os.path.basename(parsed_args.output_dir)
        pruned_file_dst = os.path.join(parsed_args.output_dir, f"{name}.cif")
        raw_file_dst = os.path.join(parsed_args.output_dir, f"{name}_raw.cif")

        os.replace(pruned_file_src, pruned_file_dst)
        os.replace(raw_file_src, raw_file_dst)

        # Entropy files
        os.makedirs(
            os.path.join(parsed_args.output_dir, "entropy_scores"),
            exist_ok=True,
        )
        pruned_es_file_src = gnn_output.replace("output.cif", "output_fixed_aa_pruned_entropy_score.cif")
        raw_es_file_src = gnn_output.replace("output.cif", "output_fixed_aa_entropy_score.cif")
        
        pruned_es_file_dst = os.path.join(parsed_args.output_dir, "entropy_scores", f"{name}.cif")
        raw_es_file_dst = os.path.join(parsed_args.output_dir, "entropy_scores", f"{name}_raw.cif")
        
        os.replace(pruned_es_file_src, pruned_es_file_dst)
        os.replace(raw_es_file_src, raw_es_file_dst)
        
        if not parsed_args.keep_intermediate_results:
            for directory in os.listdir(parsed_args.output_dir):
                if directory.startswith("gnn_output_round_") or directory == "see_alpha_output":
                    shutil.rmtree(os.path.join(parsed_args.output_dir, directory))

        print("-" * 70)
        print("ModelAngelo build has been completed successfully!")
        print("-" * 70)
        print(f"You can find your output mmCIF file here: {pruned_file_dst}")
        print("-" * 70)
        print(
            f"The raw output without pruning might be useful to show \n"
            f"some parts of the map that may be modelled, \n"
            f"but could not be automatically modelled. \n"
            f"However, the amino acid classifications are generally \n"
            f"not going to be correct. \n"
            f"You can find that here: {raw_file_dst}"
        )
        print("-" * 70)
        print(
            f"(Experimental) We now have CIF files with entropy scores \n"
            f"for each residue. These are useful for identifying regions \n"
            f"of the map that have lower confidence in the residue type \n"
            f"(i.e. amino-acid or nucleotide base) identity. \n"
            f"These files are named the same as the pruned and raw files, \n"
            f"however the bfactor column has the new entropy score instead \n"
            f"of the ModelAngelo predicted confidence, which is backbone-based. \n"
            f"These files are in: {os.path.join(parsed_args.output_dir, 'entropy_scores')}"
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
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__,
    )
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)
