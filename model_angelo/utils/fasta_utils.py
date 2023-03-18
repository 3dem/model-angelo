import os
import string
from collections import namedtuple
from itertools import islice
from typing import List, Tuple

import numpy as np
from Bio import SeqIO


FASTASequence = namedtuple("FASTASequence", field_names=["seq", "rep", "chains"])


class FASTAEmptyError(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)


def download_pdb_entry_fasta_file(pdb_entry_name, output_dir):
    import wget

    pdb_link = f"https://www.rcsb.org/fasta/entry/{pdb_entry_name}"
    wget.download(pdb_link, out=output_dir)


def is_valid_fasta_ending(fasta_path: str) -> bool:
    return (
        fasta_path.endswith(".fasta")
        or fasta_path.endswith(".fa")
        or fasta_path.endswith(".faa")
        or fasta_path.endswith(".mpfa")
    )


def read_fasta(fasta_path, auth_chains=True):
    sequences = []
    sequence_names = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description.split("|")
        if len(desc) > 0:
            sequence_names.append(desc[0].replace(">", ""))
        else:
            sequence_names.append("unknown")
        if len(desc) > 1:
            chains = desc[1].split(",")
            chains = [
                x.replace("Chains", "").replace("Chain", "").strip() for x in chains
            ]
            if auth_chains:
                for i in range(len(chains)):
                    if "auth" in chains[i]:
                        chains[i] = chains[i].split("auth")[-1].strip().replace("]", "")
        else:
            chains = ["A"]
        sequences.append(FASTASequence(str(record.seq), len(chains), chains))

    if len(sequences) == 0:
        raise FASTAEmptyError(
            f"File {fasta_path} has no parsable sequences. Please check your FASTA file."
            f"For example, this can happen if your sequences are in lower case, while"
            f"this program expects them to be uppercase."
        )

    return sequences, sequence_names


def filter_small_sequences(
    fasta_sequences: List[FASTASequence], sequence_names: List[str]
):

    filtered_sequences, filtered_sequence_names = [], []
    for sequence, sequence_name in zip(fasta_sequences, sequence_names):
        filtered_seq = FASTASequence(
            seq=remove_non_residue(sequence.seq),
            rep=sequence.seq,
            chains=sequence.chains,
        )
        if len(filtered_seq.seq) > 2:
            filtered_sequences.append(filtered_seq)
            filtered_sequence_names.append(sequence_name)
    return filtered_sequences, sequence_names


def filter_nucleotide_sequences(
    fasta_sequences: List[FASTASequence], sequence_names: List[str],
):
    filtered_sequences, filtered_sequence_names = [], []
    for sequence, sequence_name in zip(fasta_sequences, sequence_names):
        filtered_seq = FASTASequence(
            seq=sequence.seq, rep=sequence_name, chains=sequence.chains,
        )
        if len(remove_nucleotides(sequence.seq)) > 0:
            filtered_sequences.append(filtered_seq)
            filtered_sequence_names.append(sequence_name)
    return filtered_sequences, sequence_names


def split_fasta_file_into_chains(fasta_path, out_dir):
    f = open(fasta_path, "r")
    lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith(">") and len(lines[i + 1].strip()) > 2:
            new_file_name = lines[i].split("|")[0][1:] + ".fasta"
            new_file = open(os.path.join(out_dir, new_file_name), "w")

            new_file.write(lines[i])
            new_file.write(lines[i + 1])

            i += 2
        else:
            i += 1


def parse_hhr(hhr_file_path, max_num_seq=np.inf, align_sequence=False):
    # TODO need to fit alignments to original sequence using ----- characters
    sequences = []
    num_seq = 0
    for line in open(hhr_file_path, "r").readlines():
        if line.startswith("T UniRef100"):
            sequences.append((f"{num_seq}", line.split()[3]))
            num_seq += 1
    if align_sequence:
        max_seq_len = max([len(seq) for _, seq in sequences])
        aligned_sequences = [
            (desc, seq) for (desc, seq) in sequences if len(seq) == max_seq_len
        ]
        sequences = aligned_sequences
    return sequences[:max_num_seq]


# Code below is from https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    return sequence.translate(translation)


gap_delete_keys = {"-": None}
gap_translation = str.maketrans(gap_delete_keys)


def remove_gaps(sequence: str) -> str:
    return sequence.translate(gap_translation)


remove_dots_keys = {".": None}
remove_dots_translation = str.maketrans(remove_dots_keys)


def remove_dots(sequence: str) -> str:
    return sequence.translate(remove_dots_translation)


remove_nucleotide_keys = {"A": None, "C": None, "G": None, "T": None, "U": None}
remove_nucleotide_translation = str.maketrans(remove_nucleotide_keys)


def remove_nucleotides(sequence: str) -> str:
    return sequence.translate(remove_nucleotide_keys)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in islice(SeqIO.parse(filename, "fasta"), nseq)
    ]


def read_hhr(filename: str, nseq: int) -> List[Tuple[str, str]]:
    return parse_hhr(filename, nseq, align_sequence=True)


def remove_non_residue(sequence: str) -> str:
    return "".join([s for s in sequence if s in "ARNDCQEGHILKMFPSTWYVU"])


def fasta_to_unified_seq(fasta_path, auth_chains=True) -> Tuple[str, int]:
    sequences, sequence_names = read_fasta(fasta_path, auth_chains=auth_chains)
    sequences = [s.seq for s in sequences]
    unified_seq_len = sum([len(s) for s in sequences])
    unified_seq = "|||".join(sequences)
    return unified_seq, unified_seq_len


def unified_seq_to_fasta(unified_seq) -> List[FASTASequence]:
    sequences = unified_seq.split("|||")
    fasta_list = [FASTASequence(s, "null", str(i)) for (i, s) in enumerate(sequences)]
    return fasta_list


def nuc_sequence_to_purpyr(sequence: str) -> str:
    return (
        sequence.replace("A", "G")
        .replace("U", "C")
        .replace("T", "C")
        .replace("DA", "G")
        .replace("DG", "G")
        .replace("DC", "C")
        .replace("DT", "C")
    )


def trim_dots(msa: str) -> str:
    N = len(msa) - 1
    # For the leading dots
    start = 0
    while start < N:
        if msa[start] != ".":
            break
        start += 1
    if start == N:
        return ""
    end = N
    while end > 0:
        if msa[end] != ".":
            break
        end -= 1
    return msa[start : end + 1]


sequence_match = dict.fromkeys(string.ascii_uppercase)
sequence_match["-"] = None

in_seq_dict = dict.fromkeys(string.ascii_letters)


def find_match_range(msa: str) -> Tuple[int, int, int]:
    N = len(msa) - 1
    start = 0
    while start < N:
        if msa[start] in sequence_match:
            break
        start += 1
    end = N
    while end > 0:
        if msa[end] in sequence_match:
            break
        end -= 1
    num_gaps_to_start = start - len(remove_dots(remove_gaps(msa[:start])))
    return start, end, num_gaps_to_start


def split_fasta_file(fasta_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fasta_lines = open(fasta_path, "r").readlines()

    file_handle = None
    i = 0
    while i < len(fasta_lines):
        if fasta_lines[i].startswith(">"):
            if file_handle is not None:
                file_handle.close()
            file_handle = open(
                os.path.join(save_dir, fasta_lines[i][1:].split("|")[0] + ".fasta"),
                "w",
            )
            file_handle.write(fasta_lines[i])
        elif file_handle is not None:
            file_handle.write(fasta_lines[i])
        i += 1

    if file_handle is not None:
        file_handle.close()
