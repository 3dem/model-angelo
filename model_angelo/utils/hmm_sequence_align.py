from collections import namedtuple
from typing import List, Tuple

import numpy as np
import pyhmmer

from model_angelo.utils.aa_probs_to_hmm import aa_logits_to_hmm, alphabet_to_index
from model_angelo.utils.fasta_utils import (
    find_match_range,
    in_seq_dict,
    remove_dots,
    remove_non_residue,
    sequence_match,
    nuc_sequence_to_purpyr,
)
from model_angelo.utils.match_to_sequence import MatchToSequence
from model_angelo.utils.residue_constants import num_prot

HMMAlignment = namedtuple(
    "HMMAlignment",
    [
        "sequence",
        "seq_idx",
        "res_idx",
        "key_start_match",
        "key_end_match",
        "match_score",
        "hmm_output_match_sequence",
        "exists_in_sequence_mask",
    ],
)


def get_hmm_alignment(
    aa_logits: np.ndarray,
    digital_prot_sequences: List[pyhmmer.easel.DigitalSequence] = [],
    digital_rna_sequences: List[pyhmmer.easel.DigitalSequence] = [],
    digital_dna_sequences: List[pyhmmer.easel.DigitalSequence] = [],
    do_pp: bool = False,
    raw_rna_sequences: List[str] = None,
    raw_dna_sequences: List[str] = None,
    confidence: np.ndarray = None,
    base_dir: str = "/tmp",
    is_nucleotide: bool = False,
) -> HMMAlignment:
    if not is_nucleotide:
        hmm = aa_logits_to_hmm(
            aa_logits, confidence=confidence, base_dir=base_dir, alphabet_type="amino"
        )
        msas = pyhmmer.hmmer.hmmalign(
            hmm, digital_prot_sequences, all_consensus_cols=True
        )
        processed_msas = msas.alignment
        seq_idx = np.argmax(
            np.array([len(remove_non_residue(x)) for x in processed_msas])
        )
        msa_index_corr = get_msa_index_correspondence(processed_msas[seq_idx])
        index_dict = alphabet_to_index["amino"]
        match_sequence = msa_index_corr.sequence
        original_pred_seq = np.argmax(aa_logits[..., :num_prot], axis=-1)
    else:
        if do_pp:
            assert raw_rna_sequences is not None or raw_dna_sequences is not None
        has_rna_seq = len(digital_rna_sequences) > 0
        has_dna_seq = len(digital_dna_sequences) > 0
        match_type = ""
        if has_rna_seq:
            hmm_rna = aa_logits_to_hmm(
                aa_logits,
                confidence=confidence,
                base_dir=base_dir,
                alphabet_type="RNA" if not do_pp else "PP",
            )
            rna_processed_msas = pyhmmer.hmmer.hmmalign(
                hmm_rna, digital_rna_sequences, all_consensus_cols=True
            ).alignment
            rna_seq_idx = np.argmax(
                np.array([len(remove_non_residue(x)) for x in rna_processed_msas])
            )
            rna_seq_val = np.max(
                np.array([len(remove_non_residue(x)) for x in rna_processed_msas])
            )
        if has_dna_seq:
            hmm_dna = aa_logits_to_hmm(
                aa_logits,
                confidence=confidence,
                base_dir=base_dir,
                alphabet_type="DNA" if not do_pp else "PP",
            )
            dna_processed_msas = pyhmmer.hmmer.hmmalign(
                hmm_dna, digital_rna_sequences, all_consensus_cols=True
            ).alignment
            dna_seq_idx = np.argmax(
                np.array([len(remove_non_residue(x)) for x in dna_processed_msas])
            )
            dna_seq_val = np.max(
                np.array([len(remove_non_residue(x)) for x in dna_processed_msas])
            )
        if has_rna_seq and has_dna_seq:
            if rna_seq_val <= dna_seq_val:
                match_type = "RNA"
            else:
                match_type = "DNA"
        elif has_rna_seq:
            match_type = "RNA"
        elif has_dna_seq:
            match_type = "DNA"

        if match_type == "":
            made_up_match = "-" * len(aa_logits)
            msa_index_corr = get_msa_index_correspondence(made_up_match)
            index_dict = alphabet_to_index["RNA"]
            seq_idx = len(digital_prot_sequences)
            original_pred_seq = np.argmax(aa_logits[..., num_prot:], axis=-1) + num_prot
        elif match_type == "RNA":
            original_seq = None if not do_pp else raw_rna_sequences[rna_seq_idx]
            msa_index_corr = get_msa_index_correspondence(
                rna_processed_msas[rna_seq_idx], original_seq=original_seq
            )
            index_dict = alphabet_to_index["RNA"]
            seq_idx = rna_seq_idx + len(digital_prot_sequences)
            original_pred_seq = (
                np.argmax(aa_logits[..., num_prot + 4 :], axis=-1) + num_prot + 4
            )
        else:
            original_seq = None if not do_pp else raw_dna_sequences[dna_seq_idx]
            msa_index_corr = get_msa_index_correspondence(
                dna_processed_msas[dna_seq_idx], original_seq=original_seq,
            )
            index_dict = alphabet_to_index["DNA"]
            seq_idx = (
                dna_seq_idx + len(digital_prot_sequences) + len(digital_rna_sequences)
            )
            original_pred_seq = (
                np.argmax(aa_logits[..., num_prot : num_prot + 4], axis=-1) + num_prot
            )
        match_sequence = msa_index_corr.sequence

    msa_sequence = np.array(
        [index_dict[x] if x in index_dict else -1 for x in match_sequence]
    )
    new_sequence = np.where(msa_sequence != -1, msa_sequence, original_pred_seq)

    match_score = len(remove_non_residue(match_sequence)) / len(match_sequence)
    return HMMAlignment(
        sequence=new_sequence,
        seq_idx=seq_idx,
        res_idx=msa_index_corr.res_idx,
        key_start_match=msa_index_corr.key_start_match,
        key_end_match=msa_index_corr.key_end_match,
        match_score=match_score,
        hmm_output_match_sequence=match_sequence,
        exists_in_sequence_mask=msa_index_corr.exists_in_sequence_mask,
    )


def best_match_to_sequences(
    prot_sequences: List[str],
    rna_sequences: List[str],
    dna_sequences: List[str],
    chain_prot_mask: List[np.ndarray],
    chain_aa_logits: List[np.ndarray],
    chain_confidences: List[np.ndarray] = None,
    base_dir: str = "/tmp",
    do_pp: bool = True,
) -> MatchToSequence:
    alphabets = [
        pyhmmer.easel.Alphabet.amino(),
        pyhmmer.easel.Alphabet.rna(),
        pyhmmer.easel.Alphabet.dna() if not do_pp else pyhmmer.easel.Alphabet.rna(),
    ]
    digital_sequences = [
        [
            pyhmmer.easel.TextSequence(
                name=bytes(f"seq_{j}", encoding="utf-8"),
                sequence=seq if i == 0 or not do_pp else nuc_sequence_to_purpyr(seq),
            ).digitize(alphabets[i])
            for j, seq in enumerate(sequences)
        ]
        for i, sequences in enumerate([prot_sequences, rna_sequences, dna_sequences])
    ]

    if chain_confidences is None:
        chain_confidences = [None] * len(chain_aa_logits)

    (
        new_sequences,
        residue_idxs,
        sequence_idxs,
        key_start_matches,
        key_end_matches,
        match_scores,
        hmm_output_match_sequences,
        exists_in_sequence_mask,
        is_nucleotide_list,
    ) = ([], [], [], [], [], [], [], [], [])
    null_sequence_id = len(digital_sequences)
    for aa_logits, confidence, prot_mask in zip(
        chain_aa_logits, chain_confidences, chain_prot_mask
    ):
        chain_len = len(aa_logits)
        is_nucleotide = np.all(~prot_mask)
        if chain_len < 3:
            new_sequences.append(np.argmax(aa_logits, axis=-1))
            residue_idxs.append(np.arange(1, chain_len + 1))
            sequence_idxs.append(null_sequence_id)
            key_start_matches.append(1)
            key_end_matches.append(chain_len + 1)
            match_scores.append(0)
            null_sequence_id += 1
            hmm_output_match_sequences.append("-" * chain_len)
            exists_in_sequence_mask.append(np.ones(chain_len, dtype=int))
            is_nucleotide_list.append(is_nucleotide)
        else:
            hmm_alignment = get_hmm_alignment(
                aa_logits,
                digital_prot_sequences=digital_sequences[0],
                digital_rna_sequences=digital_sequences[1],
                digital_dna_sequences=digital_sequences[2],
                confidence=confidence,
                base_dir=base_dir,
                is_nucleotide=is_nucleotide,
                do_pp=do_pp,
                raw_rna_sequences=rna_sequences,
                raw_dna_sequences=dna_sequences,
            )

            new_sequences.append(hmm_alignment.sequence)
            residue_idxs.append(hmm_alignment.res_idx)
            sequence_idxs.append(hmm_alignment.seq_idx)
            key_start_matches.append(hmm_alignment.key_start_match)
            key_end_matches.append(hmm_alignment.key_end_match)
            match_scores.append(hmm_alignment.match_score)
            hmm_output_match_sequences.append(hmm_alignment.hmm_output_match_sequence)
            exists_in_sequence_mask.append(hmm_alignment.exists_in_sequence_mask)
            is_nucleotide_list.append(is_nucleotide)

    return MatchToSequence(
        new_sequences=new_sequences,
        residue_idxs=residue_idxs,
        sequence_idxs=sequence_idxs,
        key_start_matches=np.array(key_start_matches),
        key_end_matches=np.array(key_end_matches),
        match_scores=np.array(match_scores),
        hmm_output_match_sequences=hmm_output_match_sequences,
        exists_in_sequence_mask=exists_in_sequence_mask,
        is_nucleotide=is_nucleotide_list,
    )


MSAIndexCorrespondence = namedtuple(
    "MSAIndexCorrespondence",
    [
        "sequence",
        "res_idx",
        "key_start_match",
        "key_end_match",
        "exists_in_sequence_mask",
    ],
)


def get_msa_index_correspondence(
    msa: str, original_seq: str = None,
) -> MSAIndexCorrespondence:
    start_match, end_match, num_gaps_to_start = find_match_range(msa)

    idxs = []
    exists_in_sequence_mask = []
    matched_sequence = ""
    j = start_match - num_gaps_to_start + 1

    for s in msa[start_match : end_match + 1]:
        if s in sequence_match:
            if original_seq is None:
                matched_sequence += s
            else:
                matched_sequence += original_seq[j - 1] if s.isalpha() else "-"
            if s.isalpha():
                exists_in_sequence_mask.append(1)
                idxs.append(j)
            else:
                exists_in_sequence_mask.append(0)
                idxs.append(-1)  # Place-holder
        if s in in_seq_dict:
            j += 1

    idxs = np.array(idxs)

    return MSAIndexCorrespondence(
        sequence=matched_sequence,
        res_idx=idxs,
        key_start_match=idxs[idxs != -1][0],
        key_end_match=idxs[idxs != -1][-1],
        exists_in_sequence_mask=np.array(exists_in_sequence_mask, dtype=int),
    )


def fix_flanking_regions(matched_sequence, full_msa, res_idx):
    start_match, end_match = res_idx.min(), res_idx.max()
    matched_sequence_end_idx = len(matched_sequence) - 1

    i = 0
    while True:
        if i >= matched_sequence_end_idx:
            break
        if matched_sequence[i].isalpha():
            break
        i += 1
    start_flank_len = i

    i = matched_sequence_end_idx
    while True:
        if i <= 0:
            break
        if matched_sequence[i].isalpha():
            break
        i -= 1
    end_flank_len = matched_sequence_end_idx - i

    msa_remove_dots = remove_dots(full_msa)
    msa_end_idx = len(msa_remove_dots) - 1

    new_start_flank = []
    i = start_match - 1
    while start_flank_len > 0:
        if i <= 0:
            break
        if msa_remove_dots[i].isalpha():
            new_start_flank.append(msa_remove_dots[i].upper())
            start_flank_len -= 1
        i -= 1
    new_start_flank = "".join(reversed(new_start_flank))

    new_end_flank = []
    i = end_match + 1
    while end_flank_len > 0:
        if i >= msa_end_idx:
            break
        if msa_remove_dots[i].isalpha():
            new_end_flank.append(msa_remove_dots[i].upper())
            end_flank_len -= 1
        i += 1
    new_end_flank = "".join(new_end_flank)

    start_flank_len, end_flank_len = len(new_start_flank), len(new_end_flank)
    sequence_middle = matched_sequence[
        start_flank_len
        if start_flank_len > 0
        else None : -end_flank_len
        if end_flank_len > 0
        else None
    ]
    return new_start_flank + sequence_middle + new_end_flank


def sort_chains(
    match_to_sequence: MatchToSequence,
    chains,
    ca_positions,
    min_chain_len=5,
    min_match_score=0.4,
    max_dist=30,
    max_seq_gap=40,
):
    unique_seqs = np.unique(match_to_sequence.sequence_idxs)

    og_chain_lens = np.array([len(c) for c in chains])
    og_chain_starts = np.array([c[0] for c in chains])
    og_chain_ends = np.array([c[-1] for c in chains])

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    chain_start_pos = ca_positions[chain_starts]
    chain_end_pos = ca_positions[chain_ends]

    new_chain_ids = [[i] for i in range(len(chains))]

    spent_starts, spent_ends = set(), set()

    for seq in unique_seqs:
        sequence_match_idx = np.nonzero(match_to_sequence.sequence_idxs == seq)[0]

        if len(sequence_match_idx) > 1:
            dist_mat = np.linalg.norm(
                chain_start_pos[sequence_match_idx, None]
                - chain_end_pos[sequence_match_idx][None],
                axis=-1,
            )
            np.fill_diagonal(dist_mat, np.inf)
            dist_mat = np.where(dist_mat < max_dist, dist_mat, np.inf)
            dist_mat[
                match_to_sequence.match_scores[sequence_match_idx] < min_match_score
            ] = np.inf
            dist_mat[
                :, match_to_sequence.match_scores[sequence_match_idx] < min_match_score
            ] = np.inf

            seq_match_chain_lens = og_chain_lens[sequence_match_idx]
            dist_mat[seq_match_chain_lens < min_chain_len] = np.inf
            dist_mat[:, seq_match_chain_lens < min_chain_len] = np.inf

            gaps = (
                match_to_sequence.key_start_matches[sequence_match_idx, None]
                - match_to_sequence.key_end_matches[sequence_match_idx][None]
            )

            dist_mat = np.where(gaps < max_seq_gap, dist_mat, np.inf)
            dist_mat = np.where(gaps > 0, dist_mat, np.inf)

        else:
            continue

        while np.any(dist_mat != np.inf):
            chain_start_idx, chain_end_idx = np.unravel_index(
                np.argmin(dist_mat), dist_mat.shape
            )
            chain_start_match, chain_end_match = (
                sequence_match_idx[chain_start_idx],
                sequence_match_idx[chain_end_idx],
            )

            chain_start_match_reidx = np.nonzero(
                chain_starts == og_chain_starts[chain_start_match]
            )[0][0]
            chain_end_match_reidx = np.nonzero(
                chain_ends == og_chain_ends[chain_end_match]
            )[0][0]
            if chain_start_match_reidx == chain_end_match_reidx:
                dist_mat[chain_start_idx, chain_end_idx] = np.inf
                continue

            new_chain = np.concatenate(
                (chains[chain_end_match_reidx], chains[chain_start_match_reidx]), axis=0
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
            new_chain_id = (
                new_chain_ids[chain_end_match_reidx]
                + new_chain_ids[chain_start_match_reidx]
            )
            tmp_chain_ids = np.array(new_chain_ids, dtype=object)[
                chain_arange[
                    (chain_arange != chain_start_match_reidx)
                    & (chain_arange != chain_end_match_reidx)
                ]
            ].tolist()
            tmp_chain_ids.append(new_chain_id)
            new_chain_ids = tmp_chain_ids

            chain_starts = np.array([c[0] for c in chains])
            chain_ends = np.array([c[-1] for c in chains])

            spent_starts.add(chain_start_match)
            spent_ends.add(chain_end_match)

            dist_mat[chain_start_idx] = np.inf
            dist_mat[:, chain_end_idx] = np.inf

    match_to_sequence.concatenate_chains(new_chain_ids)
    return chains, match_to_sequence


def sort_chains_by_match(
    chains: List[np.ndarray], best_match_output: MatchToSequence
) -> Tuple[List[np.ndarray], MatchToSequence]:
    match_scores = np.array(best_match_output.match_scores)
    new_idxs = np.argsort(-match_scores)
    new_chains = [chains[i] for i in new_idxs]
    return (
        new_chains,
        MatchToSequence(
            new_sequences=[best_match_output.new_sequences[i] for i in new_idxs],
            residue_idxs=[best_match_output.residue_idxs[i] for i in new_idxs],
            sequence_idxs=[best_match_output.sequence_idxs[i] for i in new_idxs],
            key_start_matches=best_match_output.key_start_matches[new_idxs],
            key_end_matches=best_match_output.key_end_matches[new_idxs],
            match_scores=best_match_output.match_scores[new_idxs],
            hmm_output_match_sequences=[
                best_match_output.hmm_output_match_sequences[i] for i in new_idxs
            ],
            exists_in_sequence_mask=[
                best_match_output.exists_in_sequence_mask[i] for i in new_idxs
            ],
            is_nucleotide=[best_match_output.is_nucleotide[i] for i in new_idxs],
        ),
    )


FixChainsOutput = namedtuple(
    "FixChainsOutput", ["chains", "best_match_output", "unmodelled_sequences",],
)


def fix_chains_pipeline(
    prot_sequences: List[str],
    rna_sequences: List[str],
    dna_sequences: List[str],
    chains: List[int],
    chain_aa_logits: List[np.ndarray],
    ca_pos: np.ndarray,
    chain_prot_mask: List[np.ndarray],
    chain_confidences: List[np.ndarray] = None,
    base_dir: str = "/tmp",
) -> FixChainsOutput:
    """
    What you actually want is to get the smallest sum for the distance as well as the gap
    when you tie sequences together. The obvious thing that comes into mind is dynamic programming,
    which in O(n^2), but with the caveat that it can be made to be O(km^2),
    where k is the number of sequences and m is the max length of the sequence match.

    Oh its A*!!!! Shortest path between two nodes, where you can't go from all to all, just ones that have
    the average sequence being close or something. You find shortest path, with some penalty for how long the
    sequence is or whatever.

    """
    best_match_output = best_match_to_sequences(
        prot_sequences=prot_sequences,
        rna_sequences=rna_sequences,
        dna_sequences=dna_sequences,
        chain_aa_logits=chain_aa_logits,
        chain_prot_mask=chain_prot_mask,
        chain_confidences=chain_confidences,
        base_dir=base_dir,
    )
    chains = best_match_output.remove_duplicates(chains, ca_pos)
    chains, best_match_output = sort_chains(best_match_output, chains, ca_pos,)
    chains, best_match_output = sort_chains_by_match(chains, best_match_output)
    return FixChainsOutput(
        chains=chains, best_match_output=best_match_output, unmodelled_sequences=None,
    )


def prune_and_connect_chains(
    chains: List[int],
    best_match_output: MatchToSequence,
    ca_pos: np.ndarray,
    aggressive_pruning=False,
    chain_prune_length=4,
):
    chains = best_match_output.prune_chains(
        chains,
        chain_prune_length=chain_prune_length,
        aggressive_pruning=aggressive_pruning,
    )
    chains = best_match_output.remove_duplicates(chains, ca_pos)
    if aggressive_pruning:
        chains, best_match_output = sort_chains(best_match_output, chains, ca_pos,)
    chains, best_match_output = sort_chains_by_match(chains, best_match_output)
    return FixChainsOutput(
        chains=chains, best_match_output=best_match_output, unmodelled_sequences=None,
    )
