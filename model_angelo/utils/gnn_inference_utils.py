import argparse
from contextlib import nullcontext

import numpy as np
from scipy.spatial import cKDTree

from model_angelo.utils.affine_utils import (
    get_affine_translation,
    get_affine_rot,
    get_affine,
    init_random_affine_from_translation,
)
from model_angelo.utils.fasta_utils import fasta_to_unified_seq
from model_angelo.utils.pdb_utils import load_cas_ps_from_structure
from model_angelo.utils.protein import Protein, get_protein_empty_except
import torch
import torch.nn.functional as F

from model_angelo.utils.residue_constants import (
    num_net_torsions,
    canonical_num_residues,
)
from model_angelo.utils.torch_utils import get_module_device


def argmin_random(
    count_tensor: torch.Tensor,
    neighbours: torch.LongTensor,
    batch_size: int = 1,
    repeat_per_residue: int = 3,
):
    # We first look at the individual counts for each residue
    counts = count_tensor.clamp(max=repeat_per_residue)
    # If the proportion of clamped counts is too high, we use the full count tensor
    if torch.sum(counts == repeat_per_residue).item() / len(counts) > 0.7:
        neighbour_counts = count_tensor
    else:
        neighbour_counts = counts[neighbours].sum(dim=-1)
    rand_idxs = torch.randperm(len(neighbour_counts))
    corr_idxs = torch.arange(len(neighbour_counts))[rand_idxs]
    random_argmin = neighbour_counts[rand_idxs].argsort()[:batch_size]
    original_argmin = corr_idxs[random_argmin]
    return original_argmin


def get_neighbour_idxs(protein: Protein, k: int, idxs=None):
    # Get an initial set of pointers to neighbours for more efficient inference
    backbone_frames = protein.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    ca_positions = get_affine_translation(backbone_frames)
    kd = cKDTree(ca_positions)
    if idxs is None:
        _, init_neighbours = kd.query(ca_positions, k=k)
    else:
        _, init_neighbours = kd.query(ca_positions[idxs], k=k)
    return torch.from_numpy(init_neighbours)


def init_empty_collate_results(
    num_predicted_residues, unified_seq_len=None, device="cpu"
):
    result = {}
    result["counts"] = torch.zeros(num_predicted_residues, device=device)
    result["pred_positions"] = torch.zeros(num_predicted_residues, 3, device=device)
    result["pred_affines"] = torch.zeros(num_predicted_residues, 3, 4, device=device)
    result["pred_torsions"] = torch.zeros(
        num_predicted_residues, num_net_torsions, 2, device=device
    )
    result["aa_logits"] = torch.zeros(
        num_predicted_residues, canonical_num_residues, device=device
    )
    result["local_confidence"] = torch.zeros(num_predicted_residues, device=device)
    result["existence_mask"] = torch.zeros(num_predicted_residues, device=device)
    if unified_seq_len is not None:
        result["seq_attention_scores"] = torch.zeros(
            num_predicted_residues, unified_seq_len, device=device
        )
    return result


def get_inference_data(
        protein, grid_data, idxs, crop_length=200, num_devices: int = 1,
):
    cryo_grids = torch.from_numpy(grid_data.grid[None])  # Add channel dim
    backbone_frames = protein.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    ca_positions = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(ca_positions), dtype=int)

    batch = None
    batch_num = 1
    output_list = []
    batch_num_per_device = len(idxs) // num_devices
    for j in range(num_devices):
        if len(ca_positions) > crop_length:
            kd = cKDTree(ca_positions)
            _, picked_indices = kd.query(
                ca_positions[idxs[j * batch_num_per_device: (j + 1) * batch_num_per_device]], k=crop_length
            )
            batch_num = batch_num_per_device
            batch = torch.concat(
                [torch.ones(crop_length, dtype=torch.long) * i for i in range(batch_num)],
                dim=0,
            )
    
        output_dict = {
            "affines": torch.from_numpy(backbone_frames[picked_indices]),
            "cryo_grids": cryo_grids,
            "cryo_global_origins": torch.from_numpy(
                grid_data.global_origin.astype(np.float32)
            ),
            "cryo_voxel_sizes": torch.Tensor([grid_data.voxel_size]),
            "indices": torch.from_numpy(picked_indices),
            "prot_mask": torch.from_numpy(protein.prot_mask[picked_indices]),
            "num_nodes": len(picked_indices),
            "batch_num": batch_num,
            "batch": batch,
        }
        if protein.residue_to_lm_embedding is not None:
            output_dict["sequence"] = torch.from_numpy(protein.residue_to_lm_embedding)
        output_list.append(output_dict)
    return output_list


def update_protein_gt_frames(
    protein: Protein, update_indices: np.ndarray, update_affines: np.ndarray
) -> Protein:
    protein.rigidgroups_gt_frames[update_indices][:, 0] = update_affines
    return protein


def collate_nn_results(
    collated_results, results, indices, protein, num_pred_residues=50, offset=0,
):
    update_slice = np.s_[offset : num_pred_residues + offset]
    collated_results["counts"][indices[update_slice]] += 1
    collated_results["pred_positions"][indices[update_slice]] += results[
        "pred_positions"
    ][-1][update_slice]
    collated_results["pred_torsions"][indices[update_slice]] += F.normalize(
        results["pred_torsions"][update_slice], p=2, dim=-1
    )

    curr_pos_avg = (
        collated_results["pred_positions"][indices[update_slice]]
        / collated_results["counts"][indices[update_slice]][..., None]
    )
    collated_results["pred_affines"][indices[update_slice]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][update_slice]).cpu(), curr_pos_avg
    )
    collated_results["aa_logits"][indices[update_slice]] += results["cryo_aa_logits"][
        -1
    ][update_slice]
    collated_results["local_confidence"][indices[update_slice]] = results[
        "local_confidence_score"
    ][-1][update_slice][..., 0]
    collated_results["existence_mask"][indices[update_slice]] = results[
        "pred_existence_mask"
    ][-1][:num_pred_residues][..., 0]
    if "seq_attention_scores" in collated_results:
        collated_results["seq_attention_scores"][indices[update_slice]] += results[
            "seq_attention_scores"
        ][update_slice][..., 0]
    protein = update_protein_gt_frames(
        protein,
        indices[update_slice].numpy(),
        collated_results["pred_affines"][indices[update_slice]].numpy(),
    )
    return collated_results, protein


@torch.no_grad()
def run_inference_on_data(
    module,
    meta_batch_list,
    run_iters: int = 2,
    seq_attention_batch_size: int = 200,
    fp16: bool = False,
    using_cache: bool = False,
):
    with_seq = "sequence" in meta_batch_list[0]
    meta_input_list = []
    for data in meta_batch_list:
        affines = data["affines"]
        kwargs = {
            "positions": get_affine_translation(affines),
            "prot_mask": data["prot_mask"],
            "init_affine": affines,
            "record_training": False,
            "run_iters": run_iters,
            "using_cache": using_cache,
        }
        if with_seq:
            kwargs["seq_attention_batch_size"] = seq_attention_batch_size
        if data["batch_num"] == 1:
            if with_seq:
                kwargs["sequence"] = data["sequence"][None]
                kwargs["sequence_mask"] = torch.ones(1, data["sequence"].shape[0])
            kwargs["batch"] = None
            kwargs["cryo_grids"] = [data["cryo_grids"]]
            kwargs["cryo_global_origins"] = [data["cryo_global_origins"]]
            kwargs["cryo_voxel_sizes"] = [data["cryo_voxel_sizes"]]
        else:
            if with_seq:
                kwargs["sequence"] = (
                    data["sequence"][None].expand(data["batch_num"], -1, -1,)
                )
                kwargs["sequence_mask"] = torch.ones(
                    data["batch_num"], data["sequence"].shape[0],
                )
            kwargs["batch"] = data["batch"]
            kwargs["cryo_grids"] = [
                data["cryo_grids"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_global_origins"] = [
                data["cryo_global_origins"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_voxel_sizes"] = [
                data["cryo_voxel_sizes"] for _ in range(data["batch_num"])
            ]
        meta_input_list.append(kwargs)
    result = module(meta_input_list)
    return result


def init_protein_from_see_alpha(
    see_alpha_file: str, prot_fasta_file: str = None, skip_nucleotides: bool = False,
) -> Protein:
    ca_ps_dict = load_cas_ps_from_structure(see_alpha_file)
    ca_locations = torch.cat(
        (torch.from_numpy(ca_ps_dict["CA"]), torch.from_numpy(ca_ps_dict["P"]),), dim=0,
    ).float()
    rigidgroups_gt_frames = np.zeros((len(ca_locations), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        ca_locations
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(ca_locations), 1), dtype=np.float32)
    prot_mask = np.array(
        [True] * len(ca_ps_dict["CA"]) + [False] * len(ca_ps_dict["P"]), dtype=bool,
    )
    unified_seq, unified_seq_len = None, None
    if prot_fasta_file is not None:
        unified_seq, unified_seq_len = fasta_to_unified_seq(prot_fasta_file)
    residue_mask = prot_mask if skip_nucleotides else np.ones(len(rigidgroups_gt_exists), dtype=bool)
    return get_protein_empty_except(
        rigidgroups_gt_frames=rigidgroups_gt_frames[residue_mask],
        rigidgroups_gt_exists=rigidgroups_gt_exists[residue_mask],
        unified_seq=unified_seq,
        unified_seq_len=unified_seq_len,
        prot_mask=prot_mask[residue_mask],
    )


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
    if "seq_attention_scores" in final_results:
        final_results["seq_attention_scores"] = (
            collated_results["seq_attention_scores"]
            / collated_results["counts"][..., None]
        )

    return dict([(k, v.numpy()) for (k, v) in final_results.items()])


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "--i", required=True, help="The path to the input map")
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
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
    parser.add_argument("--refine", action="store_true", help="Run refinement program")
    parser.add_argument(
        "--batch-size", default=1, type=int, help="How many batches to run in parallel"
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 in inference")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="The voxel size that the GNN should be interpolating to."
    )
    return parser
