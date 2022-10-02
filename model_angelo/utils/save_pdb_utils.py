import glob
import os
import pickle
from typing import List, Union

import numpy as np
import pandas as pd
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.StructureBuilder import StructureBuilder

from model_angelo.utils.misc_utils import assertion_check
from model_angelo.utils.residue_constants import (
    index_to_restype_3,
    restype_name_to_atom14_names, index_to_restype_1,
)

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SEQUENCE_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEQUENCE_BASED_CHAIN_IDS = "abcdefghijklmnopqrstuvwxyz0123456789"


def number_to_base(n, b):
    """
    From https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base
    """
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def get_number_in_base_str(n, base_symbols) -> str:
    in_base = number_to_base(n, len(base_symbols))
    num_digits = len(in_base)
    return "".join(
        [
            base_symbols[n if i == num_digits - 1 else n - 1]
            for i, n in enumerate(in_base)
        ]
    )


def number_to_chain_str(number: int) -> str:
    return get_number_in_base_str(number, PDB_CHAIN_IDS)


def seq_id_and_number_to_chain_str(seq_id: int, number: int) -> str:
    seq_str = get_number_in_base_str(seq_id, SEQUENCE_IDS)
    chain_str = get_number_in_base_str(number, SEQUENCE_BASED_CHAIN_IDS)
    return seq_str + chain_str


def points_to_xyz(path_to_save, points, zyx_order=False):
    with open(path_to_save, "w") as f:
        f.write(f"{len(points)}\n")
        f.write("\n")
        for point in points:
            if not zyx_order:
                f.write(f"C {point[0]} {point[1]} {point[2]}\n")
            else:
                f.write(f"C {point[2]} {point[1]} {point[0]}\n")


def points_to_pdb(path_to_save, points):
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")
    for i, point in enumerate(points):
        struct.set_line_counter(i)
        struct.init_residue(f"ALA", " ", i, " ")
        struct.init_atom("CA", point, 0, 1, " ", "CA", "C")
    struct = struct.get_structure()
    io = MMCIFIO()
    io.set_structure(struct)
    io.save(path_to_save)


def chains_to_pdb(path_to_save, chains):
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    for i, chain in enumerate(chains, start=1):
        struct.init_chain(number_to_chain_str(i))
        for j, point in enumerate(chain):
            struct.set_line_counter(j)
            struct.init_residue(f"ALA", " ", j, " ")
            struct.init_atom("CA", point, 0, 1, " ", "CA", "C")
    struct = struct.get_structure()
    io = MMCIFIO()
    io.set_structure(struct)
    io.save(path_to_save)


def to_xyz(directory):
    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))

    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            ds = pickle.load(f)
        real_coordinates = ds["cas"]
        fake_coordinates = ds["random_coordinates"]
        name = os.path.split(pkl)[-1].split(".")[0]
        points_to_xyz(
            os.path.join(directory, name + "_real_coordinates.xyz"),
            real_coordinates,
            zyx_order=False,
        )
        points_to_xyz(
            os.path.join(directory, name + "_random_coordinates.xyz"),
            fake_coordinates,
            zyx_order=True,
        )


def atom14_to_cif(
    aatype: np.ndarray,
    atom14: np.ndarray,
    atom_mask: np.ndarray,
    path_to_save: str,
    bfactors: np.ndarray = None,
):
    if bfactors is None:
        bfactors = np.zeros(len(aatype))
    if len(bfactors.shape) > 1:
        bfactors = bfactors[:, 0]
    struct = StructureBuilder()
    curr_chain = 0

    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain(number_to_chain_str(curr_chain))

    prev_loc = atom14[0][0]
    for i in range(aatype.shape[0]):
        res_name_3 = index_to_restype_3[aatype[i]]
        bfactor = bfactors[i]
        atom_names = restype_name_to_atom14_names[res_name_3]
        res_counter = 0
        if np.linalg.norm(prev_loc - atom14[i][0]) > 5:
            curr_chain += 1
            struct.init_chain(number_to_chain_str(curr_chain))
        prev_loc = atom14[i][0]

        struct.init_residue(res_name_3, " ", i, " ")
        for atom_name, pos, mask in zip(atom_names, atom14[i], atom_mask[i]):
            if mask < 0.5:
                continue
            struct.set_line_counter(i + res_counter)
            struct.init_atom(
                name=atom_name,
                coord=pos,
                b_factor=bfactor,
                occupancy=1,
                altloc=" ",
                fullname=atom_name,
                element=atom_name[0],
            )
            res_counter += 1
    struct = struct.get_structure()
    io = MMCIFIO()
    io.set_structure(struct)
    io.save(path_to_save)


def chain_atom14_to_cif(
    aatype: List[np.ndarray],
    atom14: List[np.ndarray],
    atom_mask: List[np.ndarray],
    path_to_save: str,
    bfactors: List[np.ndarray] = None,
    sequence_idxs: Union[List, np.ndarray] = None,
    res_idxs: List[np.ndarray] = None,
):
    struct = StructureBuilder()

    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")

    if bfactors is None:
        bfactors = [np.zeros(len(chain_aas)) for chain_aas in aatype]
    if res_idxs is None:
        res_idxs = [np.arange(1, len(chain_aas) + 1) for chain_aas in aatype]

    for j in range(len(bfactors)):
        if len(bfactors[j].shape) > 1:
            bfactors[j] = bfactors[j][:, 0]

    name_with_sequences = True
    if sequence_idxs is None:
        sequence_idxs = [0 for _ in aatype]
        name_with_sequences = False

    idx_per_sequence = {}
    for seq_id in np.unique(sequence_idxs):
        idx_per_sequence[seq_id] = 0

    for chain_id in range(len(aatype)):
        seq_id = sequence_idxs[chain_id]
        if name_with_sequences:
            chain_name = seq_id_and_number_to_chain_str(
                seq_id, idx_per_sequence[seq_id]
            )
        else:
            chain_name = number_to_chain_str(idx_per_sequence[seq_id])
        idx_per_sequence[seq_id] += 1
        struct.init_chain(chain_name)

        assertion_check(
            len(aatype[chain_id]) == len(res_idxs[chain_id]),
            f"{len(aatype[chain_id])}, {len(res_idxs[chain_id])}",
        )
        for i in range(aatype[chain_id].shape[0]):
            res_name_3 = index_to_restype_3[aatype[chain_id][i]]
            bfactor = bfactors[chain_id][i]
            atom_names = restype_name_to_atom14_names[res_name_3]
            res_counter = 0

            struct.init_residue(res_name_3, " ", res_idxs[chain_id][i], " ")
            for atom_name, pos, mask in zip(
                atom_names, atom14[chain_id][i], atom_mask[chain_id][i]
            ):
                if mask < 0.5:
                    continue
                struct.set_line_counter(i + res_counter)
                struct.init_atom(
                    name=atom_name,
                    coord=pos,
                    b_factor=bfactor,
                    occupancy=1,
                    altloc=" ",
                    fullname=atom_name,
                    element=atom_name[0],
                )
                res_counter += 1

    struct = struct.get_structure()
    io = MMCIFIO()
    io.set_structure(struct)
    io.save(path_to_save)


def write_chain_report(
    path_to_save: str,
    sequence_idxs: Union[List, np.ndarray],
    bfactors: List[np.ndarray],
    match_scores: List[float],
    chain_prune_length: int = 0,
    hmm_output_match_sequences: List[str] = None,
):
    if hmm_output_match_sequences is None:
        hmm_output_match_sequences = ["" for _ in bfactors]

    report = {
        "chain_name": [],
        "pruned_chain_name": [],
        "average_confidence": [],
        "sequence_match_score": [],
        "chain_length": [],
        "sequence_idx": [],
        "hmm_output_match_sequences": [],
    }

    idx_per_sequence = {}
    for seq_id in np.unique(sequence_idxs):
        idx_per_sequence[seq_id] = 0

    for chain_id in range(len(sequence_idxs)):
        chain_len = len(bfactors[chain_id])
        seq_id = sequence_idxs[chain_id]

        report["chain_name"].append(number_to_chain_str(chain_id))
        report["average_confidence"].append(bfactors[chain_id].mean())
        report["sequence_match_score"].append(match_scores[chain_id])
        report["chain_length"].append(chain_len)
        report["sequence_idx"].append(seq_id)
        report["hmm_output_match_sequences"].append(
            hmm_output_match_sequences[chain_id]
        )

        if chain_len >= chain_prune_length:
            report["pruned_chain_name"].append(
                seq_id_and_number_to_chain_str(seq_id, idx_per_sequence[seq_id])
            )
            idx_per_sequence[seq_id] += 1
        else:
            report["pruned_chain_name"].append("pruned")

    df = pd.DataFrame.from_dict(report)
    df.to_csv(path_to_save, index=False)


def write_chain_probabilities(
    path_to_save: str,
    bfactors: List[np.ndarray],
    aa_probs: List[np.ndarray],
    chain_prune_length: int = 0,
):
    with open(path_to_save, "w") as file_handle:
        for chain_id in range(len(bfactors)):
            chain_len = len(bfactors[chain_id])

            file_handle.write(f"="*50 + "\n")
            file_handle.write(
                f"Chain id: {chain_id}\n"
            )
            if chain_len >= chain_prune_length:
                file_handle.write(f"Not pruned\n")
            else:
                file_handle.write(f"Pruned\n")
            file_handle.write(f"Chain length: {chain_len}\n")
            file_handle.write(
                "Confidence per residue:" + ",".join([str(x) for x in bfactors[chain_id]]) + "\n"
            )
            file_handle.write(
                "Amino acid probability per residue:" + "\n"
            )
            for i, aa in enumerate(index_to_restype_1):
                file_handle.write(
                    f"{aa}:" + ",".join([str(x) for x in aa_probs[chain_id][:, i]]) + "\n"
                )


if __name__ == "__main__":
    number_to_chain_str(62)
