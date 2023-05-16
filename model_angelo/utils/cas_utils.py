from collections import namedtuple
from random import shuffle

import numpy as np
import torch
import tqdm
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.spatial import cKDTree

from model_angelo.utils.misc_utils import setup_logger


def get_fit_report(input_cas, target_cas, max_dist=5, verbose=False, two_rounds=False):
    target_correspondence, input_correspondence = get_correspondence(
        input_cas, target_cas, max_dist, verbose, two_rounds=two_rounds
    )

    if len(target_correspondence) == 0:
        return (0, 0, 0)

    false_positive_count = len(
        set(range(len(input_cas))).difference(input_correspondence)
    )
    false_negative_count = len(
        set(range(len(target_cas))).difference(target_correspondence)
    )
    true_positive_count = len(input_correspondence)

    input_cas_cor = input_cas[input_correspondence]
    target_cas_cor = target_cas[target_correspondence]

    sup = SVDSuperimposer()
    sup.set(target_cas_cor, input_cas_cor)
    sup.run()

    rms = sup.get_rms()
    lddt_score = get_lddt(torch.Tensor(input_cas_cor), torch.Tensor(target_cas_cor))
    return (
        rms,
        lddt_score,
        true_positive_count / (true_positive_count + false_negative_count),
        true_positive_count / (true_positive_count + false_positive_count),
    )


def matrix_based_correspondence(input_cas, target_cas, max_dist, verbose):
    # Distance matrix: input_num x target_num
    distance_matrix = np.linalg.norm(input_cas[:, None] - target_cas[None], axis=-1)

    target_correspondence = []
    input_correspondence = []

    if verbose:
        pbar = tqdm.tqdm(total=len(input_cas))

    while True:
        idx = np.argmin(distance_matrix)
        which_input, which_target = np.unravel_index(idx, distance_matrix.shape)
        dist = distance_matrix[which_input, which_target]

        if dist > max_dist:
            break

        input_correspondence.append(which_input)
        target_correspondence.append(which_target)

        distance_matrix[which_input] = np.inf
        distance_matrix[:, which_target] = np.inf

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
        print("Done target correspondences")
    return target_correspondence, input_correspondence


def get_tmscore(c1, c2):
    """
    Returns residue TM-scores for two sets of coordinate c1 and c2 in shape (n_atoms, 3)
    Directly from https://github.com/psipred/DMPfold2/blob/master/dmpfold/train.py
    """
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, _, Vh = torch.linalg.svd(cov)
        V = Vh.transpose(-2, -1).conj()
    except RuntimeError:
        return None
    d = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))],
        ],
        device=c1.device,
    )
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    d0sq = ((1.24 * diffs.size(1) / 5 - 15.0) ** (1.0 / 3.0) - 1.8) ** 2
    tmscores = 1.0 / (1.0 + (diffs ** 2).sum(0) / d0sq)
    return tmscores


def kdtree_correspondence(input_cas, target_cas, max_dist=3, repeat=3):
    k = cKDTree(input_cas)
    idxs = list(range(len(target_cas)))
    corrs = []

    all_distances, all_indices = k.query(
        target_cas, k=10, distance_upper_bound=max_dist, workers=12
    )

    for r in range(repeat):
        corrs.append({})
        shuffle(idxs)
        for target_index in idxs:
            distances, indices = all_distances[target_index], all_indices[target_index]
            for d, i in zip(distances, indices):
                if d != np.inf and i not in corrs[-1]:
                    corrs[-1][target_index] = {"index": i, "distance": d}
                    break

    final_corrs = {}
    for target_index in idxs:
        idx_corrs = set()
        for r in corrs:
            if target_index in r:
                idx_corrs.add(r[target_index]["index"])
        if len(idx_corrs) == 1:
            final_corrs[target_index] = list(idx_corrs)[0]
    return final_corrs


def get_correspondence(
    input_cas,
    target_cas,
    max_dist=3,
    verbose=False,
    repeat=3,
    two_rounds=False,
    get_unmatched=False,
):
    # First round of kd tree correspondence
    final_corrs = kdtree_correspondence(
        input_cas, target_cas, max_dist=3, repeat=repeat
    )
    if two_rounds:
        target_correspondence, input_correspondence = (
            list(final_corrs.keys()),
            list(final_corrs.values()),
        )
        f_input_cas = input_cas[input_correspondence]
        f_target_cas = target_cas[target_correspondence]

        sup = SVDSuperimposer()
        sup.set(f_target_cas, f_input_cas)
        sup.run()

        rot, tran = sup.get_rotran()
        input_cas = np.dot(input_cas, rot) + tran

        final_corrs = kdtree_correspondence(
            input_cas, target_cas, max_dist=3, repeat=repeat
        )

    # Do matrix correspondences on everything left
    input_idxs = set(list(range(len(input_cas))))
    target_idxs = set(list(range(len(target_cas))))
    matched_input_idxs = set(list(final_corrs.values()))
    matched_target_idxs = set(list(final_corrs.keys()))
    unmatched_input_idxs = np.array(list(input_idxs.difference(matched_input_idxs)))
    unmatched_target_idxs = np.array(list(target_idxs.difference(matched_target_idxs)))

    if len(unmatched_input_idxs) < len(matched_input_idxs):
        unmatched_input_cas = input_cas[unmatched_input_idxs]
        unmatched_target_cas = target_cas[unmatched_target_idxs]

        unmatched_target_corr, unmatched_input_corr = matrix_based_correspondence(
            unmatched_input_cas,
            unmatched_target_cas,
            max_dist=max_dist,
            verbose=verbose,
        )

        for new_target_corr, new_input_corr in zip(
            unmatched_target_corr, unmatched_input_corr
        ):
            final_corrs[unmatched_target_idxs[new_target_corr]] = unmatched_input_idxs[
                new_input_corr
            ]

    matched_input_idxs = set(list(final_corrs.values()))
    matched_target_idxs = set(list(final_corrs.keys()))
    unmatched_input_idxs = np.array(list(input_idxs.difference(matched_input_idxs)))
    unmatched_target_idxs = np.array(list(target_idxs.difference(matched_target_idxs)))

    target_correspondence, input_correspondence = (
        np.array(list(final_corrs.keys())),
        np.array(list(final_corrs.values())),
    )
    if get_unmatched:
        return (
            target_correspondence,
            input_correspondence,
            unmatched_target_idxs,
            unmatched_input_idxs,
        )
    else:
        return target_correspondence, input_correspondence


ResidueCoordinateSystem = namedtuple(
    "ResidueCoordinateSystem", ["basis_matrix", "inv_basis_matrix"]
)


def get_residue_coordinate_systems(ca_positions, c_positions, n_positions):
    ca_c_vec = c_positions - ca_positions  # B x 3
    ca_n_vec = n_positions - ca_positions  # B x 3
    orth_basis = torch.linalg.qr(
        torch.stack((ca_c_vec, ca_n_vec), dim=1).transpose(2, 1)
    )[0]
    orth_basis = orth_basis.transpose(2, 1)

    # Basis 1 is normalized ca_o_vec and basis 2 is the orthogonal projection of basis 1 on basis 2
    basis_1 = orth_basis[:, 0]
    basis_2 = orth_basis[:, 1]
    basis_3 = torch.cross(basis_1, basis_2)

    basis_matrix = torch.stack((basis_1, basis_2, basis_3), dim=2)
    inv_basis_matrix = torch.linalg.inv(basis_matrix)
    return ResidueCoordinateSystem(
        basis_matrix=basis_matrix, inv_basis_matrix=inv_basis_matrix
    )


def get_lddt(input, target, cutoff=15.0):
    """
    The approximate lDDT score, based on AlphaFold2's code
    """
    input_dmat = torch.norm(input[..., None, :] - input, dim=-1, p=2)
    target_dmat = torch.norm(target[..., None, :] - target, dim=-1, p=2)
    dists_mask = (target_dmat < cutoff).float() * (1 - torch.eye(target_dmat.shape[-2]))

    dists_l1 = (target_dmat - input_dmat).abs()
    score = 0.25 * (
        (dists_l1 < 0.5).float()
        + (dists_l1 < 1.0).float()
        + (dists_l1 < 2.0).float()
        + (dists_l1 < 4.0).float()
    )
    norm = 1 / (1e-10 + torch.sum(dists_mask, dim=-1))
    score = norm * (1e-10 + torch.sum(dists_mask * score, dim=-1))
    return score


def rot_matrix_to_residue_coordinate_system(
    rot_matrix: torch.Tensor,
) -> ResidueCoordinateSystem:
    inv_basis_matrix = rot_matrix.transpose(-2, -1)
    return ResidueCoordinateSystem(
        basis_matrix=rot_matrix, inv_basis_matrix=inv_basis_matrix
    )


def transform_tensor_to_coordinate_system(
    tensor: torch.Tensor, residue_coordinate_system: ResidueCoordinateSystem
) -> torch.Tensor:
    """
    The tensor is assumed to come from the usual Euclidean coordinate system
    """
    return torch.einsum(
        "b...ac, b...c -> b...a", residue_coordinate_system.inv_basis_matrix, tensor
    )


def transform_tensor_from_coordinate_system(
    tensor: torch.Tensor, residue_coordinate_system: ResidueCoordinateSystem
) -> torch.Tensor:
    """
    The tensor is assumed to go to the usual Euclidean coordinate system
    """
    return torch.einsum(
        "b...ac, b...c -> b...a", residue_coordinate_system.basis_matrix, tensor
    )


def rotate_coordinate_system(
    rotation_matrix: torch.Tensor, residue_coordinate_system: ResidueCoordinateSystem
) -> ResidueCoordinateSystem:
    X = residue_coordinate_system.basis_matrix
    X_inv = residue_coordinate_system.inv_basis_matrix
    R_inv = rotation_matrix.transpose(-1, -2)
    return ResidueCoordinateSystem(
        basis_matrix=rotation_matrix @ X, inv_basis_matrix=X_inv @ R_inv
    )


def residue_coordinate_system_to_device(
    residue_coordinate_system: ResidueCoordinateSystem, device: str = "cuda"
) -> ResidueCoordinateSystem:
    return ResidueCoordinateSystem(
        basis_matrix=residue_coordinate_system.basis_matrix.to(device),
        inv_basis_matrix=residue_coordinate_system.inv_basis_matrix.to(device),
    )


def reshape_residue_coordinate_system(
    residue_coordinate_system: ResidueCoordinateSystem,
) -> ResidueCoordinateSystem:
    return ResidueCoordinateSystem(
        basis_matrix=residue_coordinate_system.basis_matrix.reshape(-1, 3, 3),
        inv_basis_matrix=residue_coordinate_system.inv_basis_matrix.reshape(-1, 3, 3),
    )


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    from model_angelo.utils.pdb_utils import load_cas_from_structure

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-structure",
        "--p",
        required=True,
        help="Where your prediction cif file is",
    )
    parser.add_argument(
        "--target-structure",
        "--t",
        required=True,
        help="Where your real model cif file is",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=5,
        help="In angstrom (Å), the maximum distance for correspondence",
    )
    parser.add_argument(
        "--two-rounds", action="store_true", help="Do two rounds of fitting"
    )
    parser.add_argument("--output-file", help="If set, saves the results to a file")
    args = parser.parse_args()

    if args.output_file is not None:
        dir_name = os.path.dirname(args.output_file)
        os.makedirs(dir_name, exist_ok=True)
        logger = setup_logger(args.output_file)
        print_fn = logger.info
    else:
        print_fn = print

    predicted_cas = load_cas_from_structure(args.predicted_structure, quiet=True)
    target_cas = load_cas_from_structure(args.target_structure, quiet=True)
    rmsd, lddt_score, recall, precision = get_fit_report(
        predicted_cas,
        target_cas,
        max_dist=args.max_dist,
        verbose=True,
        two_rounds=args.two_rounds,
    )

    if args.output_file is not None:
        with open(os.path.join(dir_name, "evaluation.pkl"), "wb") as p_wf:
            pickle.dump(
                {"rmsd": rmsd, "recall": recall, "precision": precision,}, p_wf,
            )

    print_fn("*" * 50)
    print_fn(
        f"Results for \nPrediction file: {args.predicted_structure}\n"
        f"Target file: {args.target_structure}\n"
        f"Maximum distance of {args.max_dist} Å are"
    )
    print_fn("*" * 50)

    print_fn(
        f"**** RMSD:       {rmsd:.3f} Å\n**** Recall:     {recall:.3f}\n**** Precision:  {precision:.3f}",
    )

    print_fn(f"**** lDDT score: {lddt_score.mean():.3f}")
