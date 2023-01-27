import os
import sys

import numpy as np
import torch
import tqdm
from loguru import logger

from model_angelo.data.generate_complete_prot_files import get_lm_embeddings_for_protein
from model_angelo.gnn.flood_fill import final_results_to_cif
from model_angelo.utils.affine_utils import (
    get_affine,
    get_affine_rot,
)
from model_angelo.utils.fasta_utils import fasta_to_unified_seq, is_valid_fasta_ending
from model_angelo.utils.gnn_inference_utils import (
    get_neighbour_idxs,
    init_empty_collate_results,
    get_inference_data,
    argmin_random,
    collate_nn_results,
    run_inference_on_data,
    init_protein_from_see_alpha,
    get_final_nn_results,
    get_base_parser,
)
from model_angelo.utils.grid import MRCObject, make_model_angelo_grid, load_mrc
from model_angelo.utils.misc_utils import (
    get_esm_model,
    abort_if_relion_abort,
    pickle_dump,
)
from model_angelo.utils.protein import (
    get_protein_from_file_path,
    load_protein_from_prot,
    dump_protein_to_prot,
)
from model_angelo.utils.torch_utils import (
    checkpoint_load_latest,
    get_model_from_file,
    get_device_names,
)
from model_angelo.models.multi_gpu_wrapper import MultiGPUWrapper



def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_angelo_output_dir = os.path.dirname(args.output_dir)

    module = get_model_from_file(os.path.join(args.model_dir, "model.py"))
    step = checkpoint_load_latest(
        args.model_dir, torch.device("cpu"), match_model=False, model=module,
    )
    logger.info(f"Loaded module from step: {step}")

    module = module.eval()
    device_names = get_device_names(args.device)
    num_devices = len(device_names)
    lang_model, alphabet = get_esm_model(args.esm_model)
    batch_converter = alphabet.get_batch_converter()

    lang_model = lang_model.eval()

    if hasattr(module, "voxel_size"):
        voxel_size = module.voxel_size
    else:
        voxel_size = 1.5

    protein = None
    if args.struct.endswith("prot"):
        protein = load_protein_from_prot(args.struct)
        if protein.residue_to_lm_embedding is None:
            protein = get_lm_embeddings_for_protein(
                lang_model, batch_converter, protein
            )
    elif args.struct.endswith("cif") or args.struct.endswith("pdb"):
        if "output" in args.struct and not args.refine:
            for seq_file in [args.protein_fasta, args.rna_fasta, args.dna_fasta]:
                if seq_file is None:
                    continue
                if not is_valid_fasta_ending(seq_file):
                    raise RuntimeError(
                        f"File {seq_file} is not a supported file format."
                    )
            protein = init_protein_from_see_alpha(args.struct, args.protein_fasta)
        else:
            protein = get_protein_from_file_path(args.struct)
        protein = get_lm_embeddings_for_protein(lang_model, batch_converter, protein)
    if protein is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")

    rna_sequences, dna_sequences = [], []
    if args.rna_fasta is not None:
        rna_unified_seq, rna_seq_len = fasta_to_unified_seq(args.rna_fasta)
        if rna_seq_len > 0:
            rna_sequences = rna_unified_seq.split("|||")
    if args.dna_fasta is not None:
        dna_unified_seq, dna_seq_len = fasta_to_unified_seq(args.dna_fasta)
        if dna_seq_len > 0:
            dna_sequences = dna_unified_seq.split("|||")

    grid_data = None
    if args.map.endswith("mrc"):
        grid_data = load_mrc(args.map, multiply_global_origin=False)
        grid_data = make_model_angelo_grid(
            grid_data.grid,
            grid_data.voxel_size,
            grid_data.global_origin,
            target_voxel_size=voxel_size,
        )
        grid_data = MRCObject(
            grid=grid_data.grid,
            voxel_size=grid_data.voxel_size,
            global_origin=np.zeros((3,), dtype=np.float32),
        )
    if grid_data is None:
        raise RuntimeError(
            f"Grid volume file {args.map} is not a supported file format."
        )

    num_res = len(protein.rigidgroups_gt_frames)

    collated_results = init_empty_collate_results(
        num_res, protein.unified_seq_len, device="cpu",
    )

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)

    # Get an initial set of pointers to neighbours for more efficient inference
    init_neighbours = get_neighbour_idxs(protein, k=args.crop_length // 4)

    with MultiGPUWrapper(module, device_names, args.fp16) as wrapper:
        while residues_left > 0:
            idxs = argmin_random(
                collated_results["counts"], init_neighbours, args.batch_size * num_devices,
            )
            data = get_inference_data(
                protein, grid_data, idxs, crop_length=args.crop_length, num_devices=num_devices,
            )
            results = run_inference_on_data(
                wrapper,
                data,
                seq_attention_batch_size=args.seq_attention_batch_size,
                fp16=args.fp16,
            )
            for device_id in range(num_devices):
                for i in range(args.batch_size):
                    collated_results, protein = collate_nn_results(
                        collated_results,
                        results[device_id],
                        data[device_id]["indices"],
                        protein,
                        offset=i * args.crop_length,
                    )
            residues_left = (
                num_res
                - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
            )
            steps_left = (
                total_steps
                - torch.sum(
                    collated_results["counts"].clip(0, args.repeat_per_residue)
                ).item()
            )

            pbar.update(n=int(steps_left_last - steps_left))
            steps_left_last = steps_left
            abort_if_relion_abort(model_angelo_output_dir)
    pbar.close()

    final_results = get_final_nn_results(collated_results)
    output_path = os.path.join(args.output_dir, "output.cif")

    # For debugging eyes only
    pickle_dump(final_results, os.path.join(args.output_dir, "final_results.pkl"))
    dump_protein_to_prot(protein, os.path.join(args.output_dir, "protein.prot"))
    pickle_dump(rna_sequences, os.path.join(args.output_dir, "rna_sequences.pkl"))
    pickle_dump(dna_sequences, os.path.join(args.output_dir, "dna_sequences.pkl"))

    final_results_to_cif(
        final_results,
        protein=protein,
        cif_path=output_path,
        rna_sequences=rna_sequences,
        dna_sequences=dna_sequences,
        verbose=True,
        print_fn=logger.info,
        aggressive_pruning=args.aggressive_pruning,
        save_hmms=args.write_hmm_profiles,
        refine=args.refine,
    )
    return output_path


if __name__ == "__main__":
    import argparse

    from model_angelo.utils.grid import load_mrc

    parser = get_base_parser()
    parser.add_argument(
        "--protein-fasta",
        "--pf",
        required=False,
        help="The path to the protein sequence file",
    )
    parser.add_argument(
        "--rna-fasta",
        "--rf",
        required=False,
        help="The path to the protein sequence file",
    )
    parser.add_argument(
        "--dna-fasta",
        "--df",
        required=False,
        help="The path to the protein sequence file",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm1b_t33_650M_UR50S",
        help="Which fair-esm model to use for inference, please make sure the GNN was trained with the same model",
    )
    parser.add_argument(
        "--aggressive-pruning",
        action="store_true",
        help="Only build parts of the model that have a good match with the sequence. "
        + "Will lower recall, but quality of build is higher",
    )
    parser.add_argument(
        "--seq-attention-batch-size",
        type=int,
        default=200,
        help="Lower memory usage by processing the sequence in batches.",
    )
    parser.add_argument(
        "--write-hmm-profiles",
        action="store_true",
        help="Write HMM profiles, even though it is built with sequence.",
    )
    args = parser.parse_args()
    infer(args)
