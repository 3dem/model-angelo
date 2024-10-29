import math
from typing import Iterable

import numpy as np
import torch

from model_angelo.utils.misc_utils import assertion_check
from model_angelo.utils.residue_constants import (
    restype_1_order_to_hmm,
    index_to_hmm_restype_1,
    num_prot,
    restype_1_to_index,
    restype_3_to_index,
)

from pyhmmer.plan7 import HMM, HMMFile
import os


def negative_log_prob_to_hmm_line(nlp: Iterable) -> str:
    return "  ".join([f"{x: >.5f}" if np.isfinite(x) else "*" for x in nlp])


fixed_insertion_log_probs = {
    "A": 2.54091,
    "C": 4.18910,
    "D": 2.92766,
    "E": 2.70561,
    "F": 3.22625,
    "G": 2.66633,
    "H": 3.77575,
    "I": 2.83006,
    "K": 2.82275,
    "L": 2.33953,
    "M": 3.73926,
    "N": 3.18354,
    "P": 3.03052,
    "Q": 3.22984,
    "R": 2.91696,
    "S": 2.68331,
    "T": 2.91750,
    "V": 2.69798,
    "W": 4.47296,
    "Y": 3.49288,
}

amino_preamble = """STATS LOCAL MSV       -9.9014  0.70957
                    STATS LOCAL VITERBI  -10.7224  0.70957
                    STATS LOCAL FORWARD   -4.1637  0.70957
                    HMM          A        C        D        E        F        G        H        I        K        L        M        N        P        Q        R        S        T        V        W        Y   
                                m->m     m->i     m->d     i->m     i->i     d->m     d->d
                    COMPO   2.36553  4.52577  2.96709  2.70473  3.20818  3.02239  3.41069  2.90041  2.55332  2.35210  3.67329  3.19812  3.45595  3.16091  3.07934  2.66722  2.85475  2.56965  4.55393  3.62921
                            2.68640  4.42247  2.77497  2.73145  3.46376  2.40504  3.72516  3.29302  2.67763  2.69377  4.24712  2.90369  2.73719  3.18168  2.89823  2.37879  2.77497  2.98431  4.58499  3.61525
                            0.57544  1.78073  1.31293  1.75577  0.18968  0.00000        *
                """

rna_preamble = """STATS LOCAL MSV      -14.4578  0.69450
                  STATS LOCAL VITERBI  -18.1894  0.69450
                  STATS LOCAL FORWARD   -6.5683  0.69450
                  HMM          A        C        G        U   
                              m->m     m->i     m->d     i->m     i->i     d->m     d->d
                  COMPO   1.14539  1.65748  1.58069  1.25370
                          1.38629  1.38629  1.38629  1.38629
                          0.57544  1.78073  1.31293  1.75577  0.18968  0.00000        *
               """

dna_preamble = """STATS LOCAL MSV      -14.8205  0.69414
                  STATS LOCAL VITERBI  -18.3741  0.69414
                  STATS LOCAL FORWARD   -7.8667  0.69414
                  HMM          A        C        G        T   
                              m->m     m->i     m->d     i->m     i->i     d->m     d->d
                  COMPO   1.39613  1.43307  1.42678  1.29539
                          1.38629  1.38629  1.38629  1.38629
                          0.57544  1.78073  1.31293  1.75577  0.18968  0.00000        *
               """
alphabet_to_preamble = {
    "amino": amino_preamble,
    "RNA": rna_preamble,
    "DNA": dna_preamble,
}
alphabet_to_slice = {
    "amino": np.s_[..., :num_prot],
    "DNA": np.s_[..., num_prot : num_prot + 4],
    "RNA": np.s_[..., num_prot + 4 :],
    "PP": np.s_[..., num_prot:],
}
alphabet_to_index = {
    "amino": restype_1_to_index,
    "RNA": {"A": 24, "C": 25, "G": 26, "U": 27},
    "DNA": {"A": 20, "C": 21, "G": 22, "T": 23},
}


def aa_log_probs_to_hmm_file(
    name: str,
    aa_log_probs: np.ndarray,
    confidence: np.ndarray = None,
    output_path: str = None,
    delta=0.05,
    gamma=0.5,
    alphabet_type="amino",
):
    """
    This outputs an HMMER3 file.
    """
    assertion_check(
        len(str(len(aa_log_probs))) <= 7,
        f"Cannot convert chain to HMM profile as it is too long",
    )
    if output_path is None:
        output_path = name + ".hmm"
    if confidence is None:
        confidence = np.ones(len(aa_log_probs))

    with open(output_path, "w") as file_handle:
        file_handle.write(
            f"""HMMER3/f [3.2 | April 2018]
                NAME  {name}
                LENG  {len(aa_log_probs)}
                ALPH  {alphabet_type}
                RF    no
                MM    no
                CONS  yes
                CS    no
                MAP   yes
                {alphabet_to_preamble[alphabet_type]} 
            """
        )

        negative_log_prob = -aa_log_probs
        negative_log_prob = negative_log_prob[alphabet_to_slice[alphabet_type]]
        if alphabet_type == "amino":
            negative_log_prob = negative_log_prob[
                :, restype_1_order_to_hmm
            ]  # Reorder to HMM order
            index_to_str = index_to_hmm_restype_1
        elif alphabet_type == "RNA":
            index_to_str = ["A", "C", "G", "U"]
        elif alphabet_type == "DNA":
            index_to_str = ["A", "C", "G", "T"]

        for res_index in range(len(aa_log_probs)):
            # Emission
            aa_prob_str = f"      {res_index + 1: >7}   "
            aa_prob_str += negative_log_prob_to_hmm_line(negative_log_prob[res_index])
            # For now, might need to replace based on RF,MM,CONS,CS,MAP
            aa_prob_str += f"{res_index + 1: >7} {index_to_str[np.argmin(negative_log_prob[res_index])].lower()} - - -\n"
            file_handle.write(aa_prob_str)
            # Inserts
            file_handle.write("        ")
            file_handle.write(
                negative_log_prob_to_hmm_line(negative_log_prob[res_index]) + "\n"
            )
            # Transitions
            mm = max(confidence[res_index] - delta, gamma)
            file_handle.write("        ")
            # m -> m
            file_handle.write(f"  {-math.log(mm): >.5f}")
            # m->i
            file_handle.write(f"  {-math.log((1. - mm) / 2.): >.5f}")
            # m->d
            file_handle.write(f"  {-math.log((1. - mm) / 2.): >.5f}")
            # i->m
            file_handle.write(f"  {-math.log(1. - delta): >.5f}")
            # i->i
            file_handle.write(f"  {-math.log(delta): >.5f}")
            # d->m
            file_handle.write(f"  {-math.log(1 - delta): >.5f}")
            # d->d
            file_handle.write(f"  {-math.log(delta): >.5f}\n")
        file_handle.write("//\n")


def aa_logits_to_hmm(
    aa_logits: np.ndarray,
    confidence: np.ndarray = None,
    base_dir: str = "/tmp",
    alphabet_type: str = "amino",
    eps: float = 1e-6,
) -> HMM:
    processed_aa_logits = np.ones_like(aa_logits) * -100
    processed_aa_logits[alphabet_to_slice[alphabet_type]] = aa_logits[
        alphabet_to_slice[alphabet_type]
    ]
    if alphabet_type != "PP":
        aa_log_probs = torch.from_numpy(processed_aa_logits).log_softmax(dim=-1).numpy()
    else:
        alphabet_type = "RNA"  # Treat all purine-pyrimidine matches as RNA
        # What follows is a custom implementation of log_softmax
        c = aa_logits[..., 20:].max(axis=-1, keepdims=True)
        m = aa_logits[..., 20:].min(axis=-1, keepdims=True)
        exp_logits = np.exp(aa_logits - c)
        exp_logits_gather = np.zeros_like(exp_logits) + np.exp(m - c)
        exp_logits_gather[..., restype_3_to_index["G"]] = exp_logits[
            ...,
            [
                restype_3_to_index["DA"],
                restype_3_to_index["DG"],
                restype_3_to_index["A"],
                restype_3_to_index["G"],
            ],
        ].sum(axis=-1)
        exp_logits_gather[..., restype_3_to_index["C"]] = exp_logits[
            ...,
            [
                restype_3_to_index["DC"],
                restype_3_to_index["DT"],
                restype_3_to_index["C"],
                restype_3_to_index["U"],
            ],
        ].sum(axis=-1)
        exp_logits_gather += eps
        aa_logits_gather = np.log(exp_logits_gather)
        logsumexp = np.log(exp_logits_gather.sum(axis=-1, keepdims=True))
        aa_log_probs = aa_logits_gather - logsumexp

    tmp_path = os.path.join(base_dir, f"model_angelo_temp.hmm")
    aa_log_probs_to_hmm_file(
        name="model_angelo_search",
        aa_log_probs=aa_log_probs,
        confidence=confidence,
        output_path=tmp_path,
        alphabet_type=alphabet_type,
    )

    with HMMFile(tmp_path) as hmm_file:
        hmm = hmm_file.read()

    os.remove(tmp_path)
    return hmm


def dump_aa_logits_to_hmm_file(
    aa_logits: np.ndarray,
    output_file: str,
    confidence: np.ndarray = None,
    name: str = "model_angelo_search",
    alphabet_type: str = "amino",
):
    aa_probs = torch.from_numpy(aa_logits).log_softmax(dim=-1).numpy()
    aa_log_probs_to_hmm_file(
        name,
        aa_probs,
        confidence=confidence,
        output_path=output_file,
        alphabet_type=alphabet_type,
    )
