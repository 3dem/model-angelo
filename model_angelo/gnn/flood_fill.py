import os
from collections import namedtuple
from typing import Dict, Callable, List
from model_angelo.utils.misc_utils import upper_and_lower_case_annotation

import numpy as np
import torch
from scipy.spatial import cKDTree

from model_angelo.utils.aa_probs_to_hmm import dump_aa_logits_to_hmm_file
from model_angelo.utils.affine_utils import get_affine_translation
from model_angelo.utils.save_pdb_utils import number_to_chain_str, protein_to_cif

from model_angelo.utils.hmm_sequence_align import (
    FixChainsOutput,
    fix_chains_pipeline,
    prune_and_connect_chains,
)
from model_angelo.utils.protein import (
    frames_and_literature_positions_to_atomc_pos,
    torsion_angles_to_frames,
    Protein,
)
from model_angelo.utils.residue_constants import (
    restype_atomc_mask,
    select_torsion_angles,
    restype3_to_atoms,
    num_prot,
)
from model_angelo.utils.save_pdb_utils import (
    chain_atom14_to_cif,
    write_chain_report,
    write_chain_probabilities,
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
    mid_point: float = 0.7,
    scale_multiplier: float = 6,
) -> np.ndarray:
    scale = worst_value - best_value
    score = scale_multiplier * (local_confidence_score - mid_point) / scale
    normalized_score = 1 / (1 + np.exp(score))
    return normalized_score


def chains_to_atoms(
    final_results: Dict,
    fix_chains_output: FixChainsOutput,
    backbone_affine,
    existence_mask,
):
    fixed_aatype_from_sequence = fix_chains_output.best_match_output.new_sequences
    chains = fix_chains_output.chains
    aa_probs = (
        torch.from_numpy(final_results["aa_logits"][existence_mask])
        .softmax(dim=-1)
        .numpy()
    )

    (
        chain_all_atoms, 
        chain_atom_mask, 
        chain_bfactors, 
        chain_entropy_scores, 
        chain_aa_probs,
    ) = ([], [], [], [], [])
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
            frames_and_literature_positions_to_atomc_pos(
                fixed_aatype_from_sequence[chain_id], all_frames
            )
        )
        chain_atom_mask.append(restype_atomc_mask[fixed_aatype_from_sequence[chain_id]])
        chain_bfactors.append(
            normalize_local_confidence_score(
                final_results["local_confidence"][existence_mask][chains[chain_id]]
            )
            * 100
        )
        chain_entropy_scores.append(
            final_results["entropy_score"][existence_mask][chains[chain_id]] * 100
        )
        chain_aa_probs.append(aa_probs[chains[chain_id]])
    return (
        chain_all_atoms,
        chain_atom_mask,
        chain_bfactors,
        chain_entropy_scores,
        chain_aa_probs,
    )


def remove_overlapping_ca(
    ca_positions: np.ndarray, radius_threshold: float = 0.5,
) -> np.ndarray:
    kdtree = cKDTree(ca_positions)
    existence_mask = np.ones(len(ca_positions), dtype=bool)

    for i in range(len(ca_positions)):
        if existence_mask[i]:
            too_close = np.array(
                kdtree.query_ball_point(ca_positions[i], r=radius_threshold,)
            )
            too_close = too_close[too_close != i]
            existence_mask[too_close] = False
    return existence_mask


def final_results_to_cif(
    final_results: dict,
    protein: Protein,
    cif_path: str,
    rna_sequences: List = None,
    dna_sequences: List = None,
    verbose: bool = False,
    print_fn: Callable = print,
    aggressive_pruning: bool = False,
    save_hmms: bool = False,
    refine: bool = False,
    chain_prune_length: int = 4,
    temperature: float = 0.5,
):
    prot_mask = protein.prot_mask
    if protein.unified_seq is not None:
        prot_sequences = protein.unified_seq.split("|||")
    else:
        prot_sequences = None
    if protein.aatype is not None:
        aatype = protein.aatype
    else:
        # Rest of code
        aatype = np.zeros((len(final_results["aa_logits"]),), dtype=np.int64)
        aatype[prot_mask] = np.argmax(
            final_results["aa_logits"][prot_mask][..., :num_prot], axis=-1
        )
        aatype[~prot_mask] = (
                np.argmax(final_results["aa_logits"][~prot_mask][..., num_prot:], axis=-1)
                + num_prot
        )
    # existence_mask = (
    #     (torch.from_numpy(final_results["existence_mask"]).sigmoid() > 0.3).numpy()
    #     if not refine
    #     else np.ones_like(final_results["existence_mask"]).astype(bool)
    # )
    existence_mask = np.ones_like(final_results["existence_mask"]).astype(bool) 
    backbone_affine = torch.from_numpy(final_results["pred_affines"])
    if not refine:
        # Remove overlapping residues
        existence_mask[prot_mask] *= remove_overlapping_ca(
            get_affine_translation(backbone_affine[prot_mask])
        )
        existence_mask[~prot_mask] *= remove_overlapping_ca(
            get_affine_translation(backbone_affine[~prot_mask])
        )
        aatype = aatype[existence_mask]
    backbone_affine = backbone_affine[existence_mask]
    final_results["aa_logits"][prot_mask][..., num_prot:] = -100
    final_results["aa_logits"][~prot_mask][..., :num_prot] = -100
    # Calculate entropy score
    exp_logits = np.exp(final_results["aa_logits"])
    logit_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    aa_entropy = -np.sum(final_results["aa_logits"] * logit_probs, axis=-1)
    final_results["entropy_score"] = local_confidence_score_sigmoid(
        - aa_entropy, best_value=5.0, worst_value=2.0, mid_point=3.0,
    )

    torsion_angles = select_torsion_angles(
        torch.from_numpy(final_results["pred_torsions"][existence_mask]), aatype=aatype
    )
    all_frames = torsion_angles_to_frames(aatype, backbone_affine, torsion_angles)
    all_atoms = frames_and_literature_positions_to_atomc_pos(aatype, all_frames)
    atom_mask = restype_atomc_mask[aatype]
    bfactors = (
        normalize_local_confidence_score(
            final_results["local_confidence"][existence_mask]
        )
        * 100
    )
    entropy_scores = final_results["entropy_score"][existence_mask] * 100

    if refine:
        protein.atomc_positions = all_atoms
        protein.b_factors = bfactors

    prot_mask = prot_mask[existence_mask]

    all_atoms_np = all_atoms.numpy()
    chains = []
    all_atom_idxs = np.arange(len(all_atoms_np))
    if refine:
        chains = [
            all_atom_idxs[protein.chain_index == i]
            for (i, _) in enumerate(protein.chain_id)
        ]
    else:
        if np.any(prot_mask):
            idxs = all_atom_idxs[prot_mask]
            prot_chains = flood_fill(
                all_atoms_np[prot_mask], bfactors[prot_mask], is_nucleotide=False
            )
            chains += [idxs[c] for c in prot_chains]
        if np.any(~prot_mask):
            idxs = all_atom_idxs[~prot_mask]
            nuc_chains = flood_fill(
                all_atoms_np[~prot_mask],
                bfactors[~prot_mask],
                is_nucleotide=True,
                n_c_distance_threshold=4,
            )
            chains += [idxs[c] for c in nuc_chains]


    if refine:
        protein_to_cif(protein, cif_path)
    else:
        chain_atom14_to_cif(
            [aatype[c] for c in chains],
            [all_atoms[c] for c in chains],
            [atom_mask[c] for c in chains],
            cif_path,
            bfactors=[bfactors[c] for c in chains],
        )
        chain_atom14_to_cif(
            [aatype[c] for c in chains],
            [all_atoms[c] for c in chains],
            [atom_mask[c] for c in chains],
            cif_path.replace(".cif", "_entropy_score.cif"),
            bfactors=[entropy_scores[c] for c in chains],
        )

    chain_aa_logits = [final_results["aa_logits"][existence_mask][c] for c in chains]
    chain_prot_mask = [prot_mask[c] for c in chains]
    chain_hmm_confidence = [
        local_confidence_score_sigmoid(
            final_results["local_confidence"][existence_mask][c]
        )
        for c in chains
    ]

    if (
        save_hmms
        or (prot_sequences is None
        and rna_sequences is None
        and dna_sequences is None)
    ):
        # Can make HMM profiles with the aa_probs
        hmm_dir_path = os.path.join(os.path.dirname(cif_path), "hmm_profiles")
        os.makedirs(hmm_dir_path, exist_ok=True)

        for i, chain_aa_logits in enumerate(chain_aa_logits):
            chain_name = (
                number_to_chain_str(i)
                if protein.chain_id is None
                else protein.chain_id[i]
            )
            # Because of stupid Windows...
            annotation = upper_and_lower_case_annotation(chain_name)
            if np.any(chain_prot_mask[i]):
                dump_aa_logits_to_hmm_file(
                    chain_aa_logits,
                    os.path.join(hmm_dir_path, f"{annotation}.hmm"),
                    name=f"{chain_name}",
                    alphabet_type="amino",
                )
            else:
                dump_aa_logits_to_hmm_file(
                    chain_aa_logits,
                    os.path.join(hmm_dir_path, f"{annotation}_rna.hmm"),
                    name=f"{chain_name}",
                    alphabet_type="RNA",
                )
                dump_aa_logits_to_hmm_file(
                    chain_aa_logits,
                    os.path.join(hmm_dir_path, f"{annotation}_dna.hmm"),
                    name=f"{chain_name}",
                    alphabet_type="DNA",
                )

    elif not refine:
        ca_pos = all_atoms_np[:, 1]

        fix_chains_output = fix_chains_pipeline(
            prot_sequences=prot_sequences,
            rna_sequences=rna_sequences,
            dna_sequences=dna_sequences,
            chains=chains,
            chain_aa_logits=chain_aa_logits,
            ca_pos=ca_pos,
            chain_prot_mask=chain_prot_mask,
            chain_confidences=chain_hmm_confidence,
            base_dir=os.path.dirname(cif_path),
        )

        (
            chain_all_atoms,
            chain_atom_mask,
            chain_bfactors,
            chain_entropy_scores,
            chain_aa_probs,
        ) = chains_to_atoms(
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
        chain_atom14_to_cif(
            fix_chains_output.best_match_output.new_sequences,
            chain_all_atoms,
            chain_atom_mask,
            cif_path.replace(".cif", "_fixed_aa_entropy_score.cif"),
            bfactors=chain_entropy_scores,
        )

        write_chain_report(
            cif_path.replace(".cif", "_chain_report.csv"),
            sequence_idxs=fix_chains_output.best_match_output.sequence_idxs,
            bfactors=chain_bfactors,
            match_scores=fix_chains_output.best_match_output.match_scores,
            chain_prune_length=chain_prune_length,
            hmm_output_match_sequences=fix_chains_output.best_match_output.hmm_output_match_sequences,
        )

        fix_chains_output = prune_and_connect_chains(
            fix_chains_output.chains,
            fix_chains_output.best_match_output,
            ca_pos,
            aggressive_pruning=aggressive_pruning,
            chain_prune_length=chain_prune_length,
        )

        (
            chain_all_atoms,
            chain_atom_mask,
            chain_bfactors,
            chain_entropy_scores,
            chain_aa_probs,
        ) = chains_to_atoms(
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
        
        chain_atom14_to_cif(
            fix_chains_output.best_match_output.new_sequences,
            chain_all_atoms,
            chain_atom_mask,
            cif_path.replace(".cif", "_fixed_aa_pruned_entropy_score.cif"),
            bfactors=chain_entropy_scores,
            sequence_idxs=fix_chains_output.best_match_output.sequence_idxs,
            res_idxs=fix_chains_output.best_match_output.residue_idxs
            if aggressive_pruning
            else None,
        )

        write_chain_probabilities(
            cif_path.replace(".cif", "_aa_probabilities.aap"),
            bfactors=chain_bfactors,
            aa_probs=chain_aa_probs,
            chain_prune_length=chain_prune_length,
        )

        if (
            verbose
            and fix_chains_output.unmodelled_sequences is not None
            and len(fix_chains_output.unmodelled_sequences) > 0
        ):
            print_fn(
                f"These sequence ids have been left unmodelled: "
                f"{fix_chains_output.unmodelled_sequences}"
            )

    return final_results


def flood_fill(
    atomc_positions, b_factors, n_c_distance_threshold=2.1, is_nucleotide=False,
):
    if is_nucleotide:
        n_idx, c_idx = (
            restype3_to_atoms["A"].index("P"),
            restype3_to_atoms["A"].index("O3'"),
        )
    else:
        n_idx, c_idx = (
            restype3_to_atoms["ALA"].index("N"),
            restype3_to_atoms["ALA"].index("C"),
        )

    n_positions = atomc_positions[:, n_idx]
    c_positions = atomc_positions[:, c_idx]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)
        possible_indices = np.array(
            kdtree.query_ball_point(
                n_positions[idx], r=n_c_distance_threshold, return_sorted=True
            )
        )
        possible_indices = possible_indices[possible_indices != idx]

        got_chain = False
        if len(possible_indices) > 0:
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

    og_chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
    og_chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

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

                chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
                chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

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
    from model_angelo.utils.protein import load_protein_from_prot
    from model_angelo.utils.misc_utils import pickle_load
    
    base_path = "/home/kjamali/Desktop/code/3dem_model_angelo/model-angelo/test/output/gnn_output_round_1"
    protein = load_protein_from_prot(
        f"{base_path}/protein.prot"
    )
    final_results = pickle_load(
        f"{base_path}/final_results.pkl"
    )
    rna_sequences = pickle_load(
        f"{base_path}/rna_sequences.pkl"
    )
    dna_sequences = pickle_load(
        f"{base_path}/dna_sequences.pkl"
    )

    final_results_to_cif(
        final_results=final_results,
        protein=protein,
        cif_path="test.cif",
        rna_sequences=rna_sequences,
        dna_sequences=dna_sequences,
        aggressive_pruning=True,
    )
