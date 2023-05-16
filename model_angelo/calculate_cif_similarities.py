import os
from typing import Tuple

from Bio.PDB import MMCIFParser

from model_angelo.utils.fasta_utils import read_fasta
import pandas as pd

_search_result_names = ["fasta_chain_id", "db_chain_id", "seq_sim", "chain_len"]
def calculate_cif_similarities(
        pdb_id: str,
        chain_sims: pd.DataFrame,
        result_dir: str = ".",
        pdb_dir: str = ".",
) -> Tuple[float, int]:
    pdb_name = pdb_id.upper()
    this_chain_sims = {}
    pdb_chain_sims = chain_sims[chain_sims.chain_name.str.contains(pdb_name)].copy()
    for _, search_result_row in pdb_chain_sims.iterrows():
        fasta_name = os.path.join(result_dir, search_result_row.chain_name + ".fasta")
        sim = search_result_row.sim
        seq, _ = read_fasta(fasta_name)
        for chain_id in seq[0].chains:
            this_chain_sims[chain_id] = sim
    pdb_file_path = os.path.join(pdb_dir, pdb_name.upper() + ".cif")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("none", pdb_file_path)
    model = list(structure.get_models())[0]
    total_len = 0
    total_sim = 0
    for chain in model:
        chain_sim = this_chain_sims.get(chain.id, 0)
        # Skip nucleotide sequences
        is_peptide = len(chain.get_unpacked_list()[0].get_resname()) > 1
        if is_peptide:
            total_len += len(chain)
            total_sim += len(chain) * chain_sim
    return total_sim / (total_len + 1), total_len


if __name__ == "__main__":
    import argparse
    import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file")
    parser.add_argument("--result-file")
    parser.add_argument("--pdb-dir")
    parser.add_argument("--output-file", default="output.csv")
    args = parser.parse_args()

    pdb_ids = open(args.input_file, "r").readlines()[0]
    chain_sims = pd.read_csv(args.result_file, names=["chain_name", "sim"])
    with open(args.output_file, "w") as f:
        f.write(f"pdb_id,seq_sim,total_len\n")
        for pdb_id in tqdm.tqdm(pdb_ids.split(",")):
            sim, total_len = calculate_cif_similarities(pdb_id, chain_sims, os.path.dirname(args.result_file), args.pdb_dir)
            f.write(f"{pdb_id},{sim},{total_len}\n")
            f.flush()
