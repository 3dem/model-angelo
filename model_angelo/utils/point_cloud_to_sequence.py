from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from model_angelo.utils.fasta_utils import FASTASequence
from model_angelo.utils.residue_constants import parse_sequence_string


def point_cloud_to_sequence_log(
    sequence: FASTASequence,
    t_to_aa_prob: Tensor,
    eps: float = 1e-6,
    device: str = "cpu",
) -> Tuple[List[int], float]:
    batch_size, T = t_to_aa_prob.shape[:2]
    # Log all probabilities
    t_to_aa_log_prob = torch.log(t_to_aa_prob + eps)
    parsed_seq = parse_sequence_string(sequence.seq)

    # B x T x N + 2 probability for each part of the sequence of matching the sequence at that idx
    # The N + 2 part is because of padding
    match_log_probs = t_to_aa_log_prob[..., parsed_seq]

    log_sum_of_middle_at_loc = F.conv2d(
        match_log_probs[:, None], torch.eye(T)[None, None].to(device)
    ).reshape(batch_size, -1)

    match_log_probs, match_idxs = torch.max(log_sum_of_middle_at_loc, dim=-1)
    return match_log_probs, match_idxs + T // 2


def point_cloud_to_sequence(
    sequence: FASTASequence, t_to_aa_prob: Tensor, device: str = "cpu",
) -> Tuple[List[int], float]:
    batch_size, T = t_to_aa_prob.shape[:2]
    parsed_seq = parse_sequence_string(sequence.seq)

    # B x T x N + 2 probability for each part of the sequence of matching the sequence at that idx
    # The N + 2 part is because of padding
    match_probs = t_to_aa_prob[..., parsed_seq]

    log_sum_of_middle_at_loc = F.conv2d(
        match_probs[:, None], torch.eye(T)[None, None].to(device) / T
    ).reshape(batch_size, -1)

    match_log_probs, match_idxs = torch.max(log_sum_of_middle_at_loc, dim=-1)
    return match_log_probs, match_idxs - T // 2


ChainSeqMatch = namedtuple(
    "ChainSeqMatch",
    ["prob", "matched_seq", "matched_chain", "is_reverse", "is_chain_longer",],
)


def match_chain_to_seq(
    chain, input_seq, reverse=False, bfactors=None,
) -> ChainSeqMatch:
    seq = input_seq if not reverse else list(reversed(input_seq))
    if len(chain) > len(seq):
        length = len(seq)
        end_idx = len(chain)
        chain_longer = True
    else:
        length = len(chain)
        end_idx = len(seq)
        chain_longer = False

    match_probs = chain[..., seq][None]

    if bfactors is not None:
        bfactors = torch.ones_like(match_probs) * bfactors[None, ..., None] + 1e-4
    else:
        bfactors = torch.ones_like(match_probs)
    match_probs = match_probs * bfactors

    conv_kernel = torch.eye(length)[None, None]
    prob_match = F.conv2d(match_probs, conv_kernel)
    weights = F.conv2d(bfactors, conv_kernel)
    prob_match = prob_match / weights

    prob = prob_match.max().item()
    match_end = end_idx + prob_match.argmax() - prob_match.reshape(-1).shape[0] + 1
    match_start = match_end - length

    if chain_longer:
        matched_seq = seq
        matched_chain = np.arange(len(chain))[match_start:match_end]
    else:
        matched_seq = seq[match_start:match_end]
        matched_chain = np.arange(len(chain))

    return ChainSeqMatch(
        prob=prob,
        matched_seq=matched_seq,
        matched_chain=matched_chain,
        is_reverse=reverse,
        is_chain_longer=chain_longer,
    )


def match_chains_to_sequences(
    sequences: List[str],
    chain_aa_logits: List[Union[torch.Tensor, np.ndarray]],
    bfactors: List[Union[torch.Tensor, np.ndarray]],
) -> Dict[str, str]:
    output = {}
    max_seq_len = max([len(s) for s in sequences])
    for (i, chain) in enumerate(chain_aa_logits):
        if len(chain) > max_seq_len:
            continue
        output[i] = {}
        bfactor = bfactors[i]
        if not torch.is_tensor(chain):
            chain = torch.from_numpy(chain)
        if not torch.is_tensor(bfactor):
            bfactor = torch.from_numpy(bfactor)
        chain = chain.softmax(dim=-1)
        for (j, seq) in enumerate(sequences):
            parsed_seq = parse_sequence_string(seq)
            # Usual case
            normal_match = match_chain_to_seq(
                chain, parsed_seq, reverse=False, bfactors=bfactor
            )
            if normal_match.is_chain_longer:
                continue
            # Flip
            # reversed_match = match_chain_to_seq(chain, parsed_seq, reverse=True)
            output[i][
                j
            ] = normal_match  # if normal_match.prob > reversed_match.prob else reversed_match

    return output


def best_match_to_sequences(
    sequences: List[str], chain_aa_logits: List[np.ndarray], bfactors: List[np.ndarray],
) -> Tuple[List[List[int]], Dict]:
    output = []
    output_dict = match_chains_to_sequences(sequences, chain_aa_logits, bfactors)
    for i in range(len(chain_aa_logits)):
        if i in output_dict:
            max_prob = 0
            best_seq = None
            for seq in output_dict[i]:
                if output_dict[i][seq].prob > max_prob:
                    max_prob = output_dict[i][seq].prob
                    best_seq = output_dict[i][seq].matched_seq
            output.append(best_seq)
        else:
            output.append(np.argmax(chain_aa_logits[i], axis=-1).tolist())
    return np.concatenate(output), output_dict


if __name__ == "__main__":
    f = FASTASequence("QLKKGLESGTVLIQFEQLYRKKPGLAITFAKLPQNLDKNRYKD", 1, ["A"])

    t_to_aa_probs = 0.9 * torch.Tensor(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ]
    )

    # t_to_aa_probs = torch.rand(10, 5, 20).softmax(dim=-1)
    match_log_probs, match_idxs = point_cloud_to_sequence(f, t_to_aa_probs)

    print(match_log_probs.shape, match_idxs.shape)
    print(match_log_probs, match_idxs)
    print(f"Output of algorithm index should be 6")
