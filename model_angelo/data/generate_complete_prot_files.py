import glob
import os

import esm
import numpy as np
import tqdm

from model_angelo.utils.apply_sequence_transformer import run_transformer_on_fasta
from model_angelo.utils.fasta_utils import FASTASequence
from model_angelo.utils.protein import (
    add_lm_embeddings_to_protein,
    dump_protein_to_prot,
    get_protein_from_file_path,
)


def get_lm_embeddings_for_protein(
    lang_model, batch_converter, protein, max_chain_length=1000
):
    sequences = protein.unified_seq.split("|||")
    sequences = [FASTASequence(seq, "", "A") for seq in sequences]
    seq_names = [str(x) for x in range(len(sequences))]
    result = run_transformer_on_fasta(
        lang_model,
        batch_converter,
        sequences,
        seq_names,
        repr_layers=[33],
        max_chain_length=max_chain_length,
    )
    lm_embeddings = np.concatenate(
        [result[s]["representations"][33].cpu().numpy() for s in seq_names], axis=0,
    )
    protein_with_lm = add_lm_embeddings_to_protein(protein, lm_embeddings)
    return protein_with_lm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", "--i", required=True, help="Input PDB/mmCIF files"
    )
    parser.add_argument("--output-path", "--o", required=True, help="Output prot files")
    parser.add_argument("--device", default="cuda:0", help="Which device to run on")
    parser.add_argument(
        "--max-chain-length",
        default=1000,
        help="Maximum chain length for the transformer",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        input_files = glob.glob(os.path.join(args.input_path, "*.cif")) + glob.glob(
            os.path.join(args.input_path, "*.pdb")
        )
    else:
        input_files = [args.input_path]
    pdbs = [os.path.split(file_name)[1].split(".")[0] for file_name in input_files]

    if os.path.isfile(args.output_path):
        output_path = os.path.split(args.output_path)[0]
    else:
        output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)
    output_files = [os.path.join(output_path, f"{pdb}.prot") for pdb in pdbs]

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(args.device)

    for input_file, output_file in tqdm.tqdm(zip(input_files, output_files)):
        try:
            protein = get_protein_from_file_path(input_file)
            new_protein = get_lm_embeddings_for_protein(
                model, batch_converter, protein, max_chain_length=args.max_chain_length
            )
            dump_protein_to_prot(new_protein, output_file)
        except Exception as e:
            print(e)
