from typing import Iterable

import numpy as np
import torch

from model_angelo.utils.misc_utils import assertion_check
from model_angelo.utils.residue_constants import restype_1_order_to_hmm, index_to_hmm_restype_1, index_to_restype_1

from pyhmmer.plan7 import HMM, HMMFile
import os


def negative_log_prob_to_hmm_line(nlp: Iterable) -> str:
    return "  ".join([f"{x:.5f}" if np.isfinite(x) else "*" for x in nlp])


def pseudocount_to_hhm_line(nlp: Iterable) -> str:
    return " ".join([f"{int(x)}" if np.isfinite(x) else "*" for x in nlp])


fixed_insertion_log_probs = {
    "A": 2.54091, "C": 4.18910, "D": 2.92766, "E": 2.70561,
    "F": 3.22625, "G": 2.66633, "H": 3.77575, "I": 2.83006,
    "K": 2.82275, "L": 2.33953, "M": 3.73926, "N": 3.18354,
    "P": 3.03052, "Q": 3.22984, "R": 2.91696, "S": 2.68331,
    "T": 2.91750, "V": 2.69798, "W": 4.47296, "Y": 3.49288,
}
fixed_insertion_str_hmm = "          " + negative_log_prob_to_hmm_line(
    [fixed_insertion_log_probs[aa] for aa in index_to_hmm_restype_1]
) + "\n"

non_terminal_fixed_transition_str_hmm = "          0.04082  3.91202  3.91202  0.51083  0.91629  0.51083  0.91629\n"
terminal_fixed_transition_str_hmm = "          0.02020  3.91202        *  0.51083  0.91629  0.00000        *\n"

non_terminal_fixed_transition_str_hhm = "       59       5644    5644    737     1322    737     1322    " \
                                        "10000   10000   10000\n"
terminal_fixed_transition_str_hhm = "       29       5644    *     737     1322    0       *     " \
                                    "10000   10000   10000\n"


def aa_probs_to_hmm_file(
    name: str,
    aa_probs: np.ndarray,
    output_path: str = None
):
    """
    This outputs an HMMER3 file.
    """
    assertion_check(
        len(str(len(aa_probs))) <= 7,
        f"Cannot convert chain to HMM profile as it is too long"
    )
    if output_path is None:
        output_path = name + ".hmm"

    with open(output_path, "w") as file_handle:
        file_handle.write(f"HMMER3/f [3.3.2 | Nov 2020]\n")
        file_handle.write(f"NAME  {name}\n")
        file_handle.write(f"LENG  {len(aa_probs)}\n")
        file_handle.write(f"ALPH  amino\n")
        file_handle.write(f"RF    no\n")
        file_handle.write(f"MM    no\n")
        file_handle.write(f"CONS  yes\n")  # Should this be highest AA prob?
        file_handle.write(f"CS    no\n")
        file_handle.write(f"MAP   no\n")
        file_handle.write(f"DATE  Wed Sep 14 14:08:43 2022\n")
        file_handle.write(f"COM   [1] hello\n")
        file_handle.write(f"NSEQ  1\n")
        file_handle.write(
            f"HMM          "
            f"A        C        D        E        F        G        "
            f"H        I        K        L        M        N        "
            f"P        Q        R        S        T        V        "
            f"W        Y\n"
        )
        file_handle.write(
            f"            m->m     m->i     m->d     i->m     i->i     d->m     d->d\n"
        )
        avg_per_amino_acid = aa_probs.mean(axis=0)
        nlp_avg_per_amino_acid = - np.log(avg_per_amino_acid)[restype_1_order_to_hmm]

        file_handle.write(
            "  COMPO   " + negative_log_prob_to_hmm_line(nlp_avg_per_amino_acid) + "\n"
        )
        file_handle.write(fixed_insertion_str_hmm)
        file_handle.write(non_terminal_fixed_transition_str_hmm)

        negative_log_prob = - np.log(aa_probs)
        negative_log_prob = negative_log_prob[:, restype_1_order_to_hmm]  # Reorder to HMM order

        for res_index in range(len(aa_probs)):
            aa_prob_str = f"      {res_index + 1}   "
            aa_prob_str += negative_log_prob_to_hmm_line(negative_log_prob[res_index])
            # For now, might need to replace based on RF,MM,CONS,CS,MAP
            aa_prob_str += f"      - {index_to_hmm_restype_1[np.argmin(negative_log_prob[res_index])].lower()} - - -\n"
            file_handle.write(aa_prob_str)
            file_handle.write(fixed_insertion_str_hmm)
            if res_index != len(aa_probs) - 1:
                file_handle.write(non_terminal_fixed_transition_str_hmm)
            else:
                file_handle.write(terminal_fixed_transition_str_hmm)
        file_handle.write("//\n")


def aa_probs_to_hhm_file(
    name: str,
    aa_probs: np.ndarray,
    output_path: str = None
):
    """
    This outputs an HHsearch 1.5 file.
    """
    assertion_check(
        len(str(len(aa_probs))) <= 7,
        f"Cannot convert chain to HMM profile as it is too long"
    )
    if output_path is None:
        output_path = name + ".hhm"

    with open(output_path, "w") as file_handle:
        file_handle.write(f"HHsearch 1.5\n")
        file_handle.write(f"NAME  {name}\n")
        file_handle.write(f"FILE  {name}\n")
        file_handle.write(f"LENG  {len(aa_probs)} match states, 0 columns in multiple alignment \n")
        file_handle.write(f"DATE  Wed Sep 14 14:08:43 2022\n")
        file_handle.write(f"COM   [1] hello\n")
        file_handle.write(f"NEFF  4.0 \n")
        file_handle.write(f"SEQ \n")
        file_handle.write(f">{name} \n")
        file_handle.write("".join(np.array(index_to_restype_1)[np.argmax(aa_probs, axis=-1)]) + "\n")
        file_handle.write("#\n")

        avg_per_amino_acid = aa_probs.mean(axis=0)
        pc_avg_per_amino_acid = np.round(-1000 * np.log2(avg_per_amino_acid)[restype_1_order_to_hmm])
        file_handle.write(
            "NULL   " + pseudocount_to_hhm_line(pc_avg_per_amino_acid) + "\n"
        )
        file_handle.write(
            f"HMM   "
            f"A    C    D    E    F    G    H    I    K    "
            f"L    M    N    P    Q    R    S    T    V    "
            f"W    Y\n"
        )
        file_handle.write(
            f"      M->M M->I M->D I->M I->I D->M D->D Neff NeffI NeffD\n"
        )
        file_handle.write(
            f"       0        *       *       0       *       0       *       *       *       *\n"
        )

        psuedocounts = np.round(-1000 * np.log2(aa_probs))
        psuedocounts = psuedocounts[:, restype_1_order_to_hmm]  # Reorder to HMM order
        # neff_values = np.round(np.exp(-np.sum(aa_probs * np.log(aa_probs), axis=-1)))

        for res_index in range(len(aa_probs)):
            aa_prob_str = f"{index_to_restype_1[np.argmax(aa_probs[res_index])]} {res_index + 1}   "
            aa_prob_str += pseudocount_to_hhm_line(psuedocounts[res_index])
            aa_prob_str += f" 10000   10000   10000 \n"
            file_handle.write(aa_prob_str)
            if res_index != len(aa_probs) - 1:
                file_handle.write(non_terminal_fixed_transition_str_hhm)
            else:
                file_handle.write(terminal_fixed_transition_str_hhm)
        file_handle.write("//\n")


def aa_logits_to_hmm(aa_logits: np.ndarray, base_dir: str = "/tmp") -> HMM:
    aa_probs = torch.from_numpy(aa_logits).softmax(dim=-1).numpy()
    tmp_path = os.path.join(base_dir, f"model_angelo_temp.hmm")
    aa_probs_to_hmm_file("model_angelo_search", aa_probs, tmp_path)

    with HMMFile(tmp_path) as hmm_file:
        hmm = hmm_file.read()

    os.remove(tmp_path)
    return hmm


def dump_aa_logits_to_hmm_file(
    aa_logits: np.ndarray,
    output_file: str,
    name: str = "model_angelo_search"
):
    aa_probs = torch.from_numpy(aa_logits).softmax(dim=-1).numpy()
    aa_probs_to_hmm_file(name, aa_probs, output_file)


def dump_aa_logits_to_hhm_file(
    aa_logits: np.ndarray,
    output_file: str,
    name: str = "model_angelo_search"
):
    aa_probs = torch.from_numpy(aa_logits).softmax(dim=-1).numpy()
    aa_probs_to_hhm_file(name, aa_probs, output_file)
