import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from scipy.spatial import cKDTree

from model_angelo.data.generate_complete_prot_files import get_lm_embeddings_for_protein
from model_angelo.gnn.flood_fill import final_results_to_cif
from model_angelo.utils.affine_utils import (
    get_affine,
    get_affine_rot,
    get_affine_translation,
    init_random_affine_from_translation,
)
from model_angelo.utils.fasta_utils import fasta_to_unified_seq, is_valid_fasta_ending
from model_angelo.utils.grid import MRCObject, make_model_angelo_grid, load_mrc
from model_angelo.utils.misc_utils import get_esm_model, abort_if_relion_abort, pickle_dump
from model_angelo.utils.pdb_utils import load_cas_ps_from_structure
from model_angelo.utils.protein import (
    Protein,
    get_protein_empty_except,
    get_protein_from_file_path,
    load_protein_from_prot, dump_protein_to_prot,
)
from model_angelo.utils.residue_constants import num_net_torsions, canonical_num_residues
from model_angelo.utils.torch_utils import (
    checkpoint_load_latest,
    get_model_from_file,
    get_module_device,
)


def init_protein_from_see_alpha(
        see_alpha_file: str,
        prot_fasta_file: str,
) -> Protein:
    ca_ps_dict = load_cas_ps_from_structure(see_alpha_file)
    ca_locations = torch.cat(
        (
            torch.from_numpy(ca_ps_dict["CA"]),
            torch.from_numpy(ca_ps_dict["P"]),
        ),
        dim=0,
    ).float()
    rigidgroups_gt_frames = np.zeros((len(ca_locations), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        ca_locations
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(ca_locations), 1), dtype=np.float32)
    prot_mask = np.array(
        [True] * len(ca_ps_dict["CA"]) + [False] * len(ca_ps_dict["P"]),
        dtype=bool,
    )
    unified_seq, unified_seq_len = fasta_to_unified_seq(prot_fasta_file)
    return get_protein_empty_except(
        rigidgroups_gt_frames=rigidgroups_gt_frames,
        rigidgroups_gt_exists=rigidgroups_gt_exists,
        unified_seq=unified_seq,
        unified_seq_len=unified_seq_len,
        prot_mask=prot_mask,
    )


def argmin_random(count_tensor: torch.Tensor):
    rand_idxs = torch.randperm(len(count_tensor))
    corr_idxs = torch.arange(len(count_tensor))[rand_idxs]
    random_argmin = count_tensor[rand_idxs].argmin()
    original_argmin = corr_idxs[random_argmin]
    return original_argmin


def update_protein_gt_frames(
    protein: Protein, update_indices: np.ndarray, update_affines: np.ndarray
) -> Protein:
    protein.rigidgroups_gt_frames[update_indices][:, 0] = update_affines
    return protein


def get_inference_data(protein, grid_data, idx, crop_length=200):
    grid = ((grid_data.grid - np.mean(grid_data.grid)) / np.std(grid_data.grid)).astype(
        np.float32
    )

    backbone_frames = protein.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    ca_positions = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(ca_positions), dtype=int)
    if len(ca_positions) > crop_length:
        random_res_index = idx
        kd = cKDTree(ca_positions)
        _, picked_indices = kd.query(ca_positions[random_res_index], k=crop_length)

    return {
        "affines": torch.from_numpy(backbone_frames[picked_indices]),
        "cryo_grids": torch.from_numpy(grid[None]),  # Add channel dim
        "sequence": torch.from_numpy(np.copy(protein.residue_to_lm_embedding)),
        "cryo_global_origins": torch.from_numpy(
            grid_data.global_origin.astype(np.float32)
        ),
        "cryo_voxel_sizes": torch.Tensor([grid_data.voxel_size]),
        "indices": torch.from_numpy(picked_indices),
        "prot_mask": torch.from_numpy(protein.prot_mask[picked_indices]),
        "num_nodes": len(picked_indices),
    }


@torch.no_grad()
def run_inference_on_data(
    module,
    data,
    run_iters: int = 2,
    seq_attention_batch_size: int = 200,
):
    device = get_module_device(module)

    affines = data["affines"].to(device)
    result = module(
        sequence=data["sequence"][None].to(device),
        sequence_mask=torch.ones(1, data["sequence"].shape[0], device=device),
        positions=get_affine_translation(affines),
        batch=None,
        cryo_grids=[data["cryo_grids"].to(device)],
        cryo_global_origins=[data["cryo_global_origins"].to(device)],
        cryo_voxel_sizes=[data["cryo_voxel_sizes"].to(device)],
        prot_mask=data["prot_mask"].to(device),
        init_affine=affines,
        record_training=False,
        run_iters=run_iters,
        seq_attention_batch_size=seq_attention_batch_size,
    )
    result.to("cpu")
    return result


def init_empty_collate_results(num_predicted_residues, unified_seq_len, device="cpu"):
    result = {}
    result["counts"] = torch.zeros(num_predicted_residues, device=device)
    result["pred_positions"] = torch.zeros(num_predicted_residues, 3, device=device)
    result["pred_affines"] = torch.zeros(num_predicted_residues, 3, 4, device=device)
    result["pred_torsions"] = torch.zeros(num_predicted_residues, num_net_torsions, 2, device=device)
    result["aa_logits"] = torch.zeros(num_predicted_residues, canonical_num_residues, device=device)
    result["local_confidence"] = torch.zeros(num_predicted_residues, device=device)
    result["existence_mask"] = torch.zeros(num_predicted_residues, device=device)
    result["seq_attention_scores"] = torch.zeros(
        num_predicted_residues, unified_seq_len, device=device
    )
    return result


def collate_nn_results(
    collated_results, results, indices, protein, crop_length=200, num_pred_residues=50
):
    collated_results["counts"][indices[:num_pred_residues]] += 1
    collated_results["pred_positions"][indices[:num_pred_residues]] += results[
        "pred_positions"
    ][-1][:num_pred_residues]
    collated_results["pred_torsions"][indices[:num_pred_residues]] += F.normalize(
        results["pred_torsions"][:num_pred_residues], p=2, dim=-1
    )

    curr_pos_avg = (
        collated_results["pred_positions"][indices[:num_pred_residues]]
        / collated_results["counts"][indices[:num_pred_residues]][..., None]
    )
    collated_results["pred_affines"][indices[:num_pred_residues]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][:num_pred_residues]).cpu(),
        curr_pos_avg
    )
    collated_results["aa_logits"][indices[:num_pred_residues]] += results[
        "cryo_aa_logits"
    ][-1][:num_pred_residues]
    collated_results["local_confidence"][indices[:num_pred_residues]] = results[
        "local_confidence_score"
    ][-1][:num_pred_residues][..., 0]
    collated_results["existence_mask"][indices[:num_pred_residues]] = results[
        "pred_existence_mask"
    ][-1][:num_pred_residues][..., 0]
    collated_results["seq_attention_scores"][indices[:num_pred_residues]] += results[
        "seq_attention_scores"
    ][:num_pred_residues][..., 0]

    protein = update_protein_gt_frames(
        protein,
        indices[:num_pred_residues].numpy(),
        collated_results["pred_affines"][indices[:num_pred_residues]].numpy(),
    )
    return collated_results, protein


def get_final_nn_results(collated_results):
    final_results = {}

    final_results["pred_positions"] = (
        collated_results["pred_positions"] / collated_results["counts"][..., None]
    )
    final_results["pred_torsions"] = (
        collated_results["pred_torsions"] / collated_results["counts"][..., None, None]
    )
    final_results["pred_affines"] = get_affine(
        get_affine_rot(collated_results["pred_affines"]),
        final_results["pred_positions"],
    )
    final_results["aa_logits"] = (
        collated_results["aa_logits"] / collated_results["counts"][..., None]
    )
    final_results["seq_attention_scores"] = (
        collated_results["seq_attention_scores"] / collated_results["counts"][..., None]
    )
    final_results["local_confidence"] = collated_results["local_confidence"]
    final_results["existence_mask"] = collated_results["existence_mask"]

    final_results["raw_aa_entropy"] = (
        final_results["aa_logits"].softmax(dim=-1).log().sum(dim=-1)
    )
    final_results["normalized_aa_entropy"] = final_results["raw_aa_entropy"].add(
        -final_results["raw_aa_entropy"].min()
    )
    final_results["normalized_aa_entropy"] = final_results["normalized_aa_entropy"].div(
        final_results["normalized_aa_entropy"].max()
    )

    return dict([(k, v.numpy()) for (k, v) in final_results.items()])


def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_angelo_output_dir = os.path.dirname(args.output_dir)

    module = get_model_from_file(os.path.join(args.model_dir, "model.py"))
    step = checkpoint_load_latest(
        args.model_dir,
        torch.device("cpu"),
        match_model=False,
        model=module,
    )
    logger.info(f"Loaded module from step: {step}")

    module = module.eval().to(args.device)
    lang_model, alphabet = get_esm_model(args.esm_model)
    batch_converter = alphabet.get_batch_converter()

    lang_model = lang_model.eval()

    protein = None
    if args.struct.endswith("prot"):
        protein = load_protein_from_prot(args.struct)
        if protein.residue_to_lm_embedding is None:
            protein = get_lm_embeddings_for_protein(
                lang_model, batch_converter, protein
            )
    elif args.struct.endswith("cif") or args.struct.endswith("pdb"):
        if "output" in args.struct:
            for seq_file in [args.protein_fasta, args.rna_fasta, args.dna_fasta]:
                if seq_file is None:
                    continue
                if not is_valid_fasta_ending(seq_file):
                    raise RuntimeError(f"File {seq_file} is not a supported file format.")
            protein = init_protein_from_see_alpha(args.struct, args.protein_fasta)
        else:
            protein = get_protein_from_file_path(args.struct)
        protein = get_lm_embeddings_for_protein(lang_model, batch_converter, protein)
    if protein is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")

    grid_data = None
    if args.map.endswith("mrc"):
        grid_data = load_mrc(args.map, multiply_global_origin=False)
        grid_data = make_model_angelo_grid(
            grid_data.grid,
            grid_data.voxel_size,
            grid_data.global_origin,
            target_voxel_size=1.5,
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
        num_res,
        protein.unified_seq_len,
        device="cpu",
    )

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)
    while residues_left > 0:
        idx = argmin_random(collated_results["counts"])
        data = get_inference_data(protein, grid_data, idx, crop_length=args.crop_length)
        results = run_inference_on_data(
            module, data, seq_attention_batch_size=args.seq_attention_batch_size
        )
        collated_results, protein = collate_nn_results(
            collated_results,
            results,
            data["indices"],
            protein,
            crop_length=args.crop_length,
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
    # pickle_dump(final_results, "test.pkl")

    rna_sequences, dna_sequences = [], []
    if args.rna_fasta is not None:
        rna_unified_seq, rna_seq_len = fasta_to_unified_seq(args.rna_fasta)
        if rna_seq_len > 0:
            rna_sequences = rna_unified_seq.split("|||")
    if args.dna_fasta is not None:
        dna_unified_seq, dna_seq_len = fasta_to_unified_seq(args.dna_fasta)
        if dna_seq_len > 0:
            dna_sequences = dna_unified_seq.split("|||")

    # dump_protein_to_prot(protein, "test.prot")
    # pickle_dump(rna_sequences, "rna_seq.pkl")
    # pickle_dump(dna_sequences, "dna_seq.pkl")

    final_results_to_cif(
        final_results,
        prot_mask=protein.prot_mask,
        cif_path=output_path,
        prot_sequences=protein.unified_seq.split("|||"),
        rna_sequences=rna_sequences,
        dna_sequences=dna_sequences,
        verbose=True,
        print_fn=logger.info,
        aggressive_pruning=args.aggressive_pruning,
    )

    return output_path


if __name__ == "__main__":
    import argparse

    from model_angelo.utils.grid import load_mrc

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "--i", required=True, help="The path to the input map")
    parser.add_argument(
        "--protein-fasta", "--pf", required=False, help="The path to the protein sequence file"
    )
    parser.add_argument(
        "--rna-fasta", "--rf", required=False, help="The path to the protein sequence file"
    )
    parser.add_argument(
        "--dna-fasta", "--df", required=False, help="The path to the protein sequence file"
    )
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
    parser.add_argument(
        "--dont-add-noise",
        action="store_true",
        help="Don't add noise to input protein backbone",
    )
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=200, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=1,
        type=int,
        help="How many times to repeat per residue",
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
    args = parser.parse_args()
    infer(args)
