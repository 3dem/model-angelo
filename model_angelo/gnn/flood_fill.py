import os
from collections import namedtuple
from typing import Dict

import numpy as np
import torch
from scipy.spatial import cKDTree

from model_angelo.utils.aa_probs_to_hmm import dump_aa_logits_to_hhm_file, dump_aa_logits_to_hmm_file
from model_angelo.utils.save_pdb_utils import number_to_chain_str

from model_angelo.utils.hmm_sequence_align import (
    FixChainsOutput,
    fix_chains_pipeline,
    prune_and_connect_chains,
)
from model_angelo.utils.protein import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from model_angelo.utils.residue_constants import restype_atom14_mask, select_torsion_angles
from model_angelo.utils.save_pdb_utils import (
    atom14_to_cif,
    chain_atom14_to_cif,
    write_chain_report, write_chain_probabilities,
)

FloodFillChain = namedtuple("FloodFillChain", ["start_N", "end_C", "residues"])


def normalize_local_confidence_score(
    local_confidence_score: np.ndarray,
    best_value: float = 0.5,
    worst_value: float = 1.2,
) -> np.ndarray:
    normalized_score = (worst_value - local_confidence_score) / (
        worst_value - best_value
    )
    normalized_score = np.clip(normalized_score, 0, 1)
    return normalized_score


def local_confidence_score_sigmoid(
    local_confidence_score: np.ndarray,
    best_value: float = 0.5,
    worst_value: float = 1.2,
) -> np.ndarray:
    scale = worst_value - best_value
    x_naught = scale / 2
    score = x_naught - local_confidence_score
    normalized_score = 1 / (1 + np.exp(score / scale))
    return normalized_score


def chains_to_atoms(
    final_results: Dict,
    fix_chains_output: FixChainsOutput,
    backbone_affine,
    existence_mask,
):
    fixed_aatype_from_sequence = fix_chains_output.best_match_output.new_sequences
    chains = fix_chains_output.chains
    aa_probs = torch.from_numpy(final_results["aa_logits"][existence_mask]).softmax(dim=-1).numpy()

    (
        chain_all_atoms, chain_atom_mask, chain_bfactors, chain_aa_probs,
    ) = (
        [],
        [],
        [],
        [],
    )
    # Everything below is in the order of chains
    for chain_id in range(len(chains)):
        chain_id_backbone_affine = backbone_affine[chains[chain_id]]
        torsion_angles = select_torsion_angles(
            torch.from_numpy(final_results["pred_torsions"][existence_mask])[
                chains[chain_id]
            ],
            aatype=fixed_aatype_from_sequence[chain_id],
        )
        all_frames = torsion_angles_to_frames(
            fixed_aatype_from_sequence[chain_id],
            chain_id_backbone_affine,
            torsion_angles,
        )
        chain_all_atoms.append(
            frames_and_literature_positions_to_atom14_pos(
                fixed_aatype_from_sequence[chain_id], all_frames
            )
        )
        chain_atom_mask.append(
            restype_atom14_mask[fixed_aatype_from_sequence[chain_id]]
        )
        chain_bfactors.append(
            normalize_local_confidence_score(
                final_results["local_confidence"][existence_mask][chains[chain_id]]
            )
            * 100
        )
        chain_aa_probs.append(
            aa_probs[chains[chain_id]]
        )
    return (
        chain_all_atoms,
        chain_atom_mask,
        chain_bfactors,
        chain_aa_probs,
    )


def final_results_to_cif(
    final_results,
    cif_path,
    sequences=None,
    aatype=None,
    verbose=False,
    print_fn=print,
    aggressive_pruning=False,
):
    """
    Currently assumes the ordering it comes with, I will change this later
    """
    existence_mask = (
        torch.from_numpy(final_results["existence_mask"]).sigmoid() > 0.3
    ).numpy()
    if aatype is None:
        aatype = np.argmax(final_results["aa_logits"], axis=-1)[existence_mask]
    backbone_affine = torch.from_numpy(final_results["pred_affines"])[existence_mask]
    torsion_angles = select_torsion_angles(
        torch.from_numpy(final_results["pred_torsions"][existence_mask]), aatype=aatype
    )
    all_frames = torsion_angles_to_frames(aatype, backbone_affine, torsion_angles)
    all_atoms = frames_and_literature_positions_to_atom14_pos(aatype, all_frames)
    atom_mask = restype_atom14_mask[aatype]
    bfactors = (
        normalize_local_confidence_score(
            final_results["local_confidence"][existence_mask]
        )
        * 100
    )

    all_atoms_np = all_atoms.numpy()
    chains = flood_fill(all_atoms_np, bfactors)
    chains_concat = np.concatenate(chains)

    atom14_to_cif(
        aatype[chains_concat],
        all_atoms[chains_concat],
        atom_mask[chains_concat],
        cif_path,
        bfactors=bfactors[chains_concat],
    )

    new_final_results = dict(
        [(k, v[chains_concat]) for (k, v) in final_results.items()]
    )
    new_final_results["chain_aa_logits"] = [
        final_results["aa_logits"][existence_mask][c] for c in chains
    ]
    new_final_results["hmm_confidence"] = [
        local_confidence_score_sigmoid(
            final_results["local_confidence"][existence_mask][c]
        ) for c in chains
    ]

    if sequences is None:
        # Can make HMM profiles with the aa_probs
        hmm_dir_path = os.path.join(os.path.dirname(cif_path), "hmm_profiles")
        os.makedirs(hmm_dir_path, exist_ok=True)

        for i, chain_aa_logits in enumerate(new_final_results["chain_aa_logits"]):
            chain_name = number_to_chain_str(i)
            dump_aa_logits_to_hmm_file(
                chain_aa_logits,
                os.path.join(hmm_dir_path, f"{chain_name}.hmm"),
                name=f"{chain_name}",
            )
    else:
        ca_pos = all_atoms_np[:, 1]

        fix_chains_output = fix_chains_pipeline(
            sequences,
            chains,
            new_final_results["chain_aa_logits"],
            ca_pos,
            chain_confidences=new_final_results["hmm_confidence"],
            base_dir=os.path.dirname(cif_path),
        )

        chain_all_atoms, chain_atom_mask, chain_bfactors, chain_aa_probs = chains_to_atoms(
            final_results, fix_chains_output, backbone_affine, existence_mask
        )

        for chain_id, chain in enumerate(fix_chains_output.chains):
            ca_pos[chain] = chain_all_atoms[chain_id][:, 1]

        chain_atom14_to_cif(
            fix_chains_output.best_match_output.new_sequences,
            chain_all_atoms,
            chain_atom_mask,
            cif_path.replace(".cif", "_fixed_aa.cif"),
            bfactors=chain_bfactors,
        )

        write_chain_report(
            cif_path.replace(".cif", "_chain_report.csv"),
            sequence_idxs=fix_chains_output.best_match_output.sequence_idxs,
            bfactors=chain_bfactors,
            match_scores=fix_chains_output.best_match_output.match_scores,
            chain_prune_length=4,
            hmm_output_match_sequences=fix_chains_output.best_match_output.hmm_output_match_sequences,
        )

        fix_chains_output = prune_and_connect_chains(
            fix_chains_output.chains,
            fix_chains_output.best_match_output,
            ca_pos,
            aggressive_pruning=aggressive_pruning,
            chain_prune_length=4,
        )

        chain_all_atoms, chain_atom_mask, chain_bfactors, chain_aa_probs = chains_to_atoms(
            final_results, fix_chains_output, backbone_affine, existence_mask
        )

        chain_atom14_to_cif(
            fix_chains_output.best_match_output.new_sequences,
            chain_all_atoms,
            chain_atom_mask,
            cif_path.replace(".cif", "_fixed_aa_pruned.cif"),
            bfactors=chain_bfactors,
            sequence_idxs=fix_chains_output.best_match_output.sequence_idxs,
            res_idxs=fix_chains_output.best_match_output.residue_idxs
            if aggressive_pruning
            else None,
        )

        write_chain_probabilities(
            cif_path.replace(".cif", "_aa_probabilities.aap"),
            bfactors=chain_bfactors,
            aa_probs=chain_aa_probs,
            chain_prune_length=4,
        )

        if (
            verbose
            and fix_chains_output.unmodelled_sequences is not None
            and len(fix_chains_output.unmodelled_sequences) > 0
        ):
            print_fn(
                f"These sequence ids have been left unmodelled: {fix_chains_output.unmodelled_sequences}"
            )

    return new_final_results


def flood_fill(atom14_positions, b_factors, n_c_distance_threshold=2.1):
    n_positions = atom14_positions[:, 0]
    c_positions = atom14_positions[:, 2]
    n_c_distances = np.linalg.norm(n_positions[:, None] - c_positions[None], axis=-1)
    b_factors_copy = np.copy(b_factors)
    idxs = np.arange(len(atom14_positions))

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)
        possible_edges = (n_c_distances[idx] < n_c_distance_threshold) * (
            n_c_distances[idx] > 0
        )
        got_chain = False
        if np.sum(possible_edges) > 0:
            idx_n_c_distances = n_c_distances[idx][possible_edges]
            possible_indices = idxs[possible_edges]

            sorted_indices = np.argsort(idx_n_c_distances)
            possible_indices = possible_indices[sorted_indices]

            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains])
    og_chain_ends = np.array([c[-1] for c in chains])

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=n_c_distance_threshold, return_sorted=True
        )
        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains])
                chain_ends = np.array([c[-1] for c in chains])

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains


if __name__ == "__main__":
    from model_angelo.utils.fasta_utils import read_fasta
    from model_angelo.utils.misc_utils import pickle_load

    f = pickle_load("/home/kjamali/Downloads/sofia_struct/nn_output_dict.pkl")
    seq = read_fasta("/home/kjamali/Downloads/sofia_struct/sequence.fasta")
    seq_list = [x.seq for x in seq[0]]
    final_results = final_results_to_cif(
        f,
        "/home/kjamali/Downloads/sofia_struct/testing2.cif",
        seq_list,
        aggressive_pruning=True,
    )
