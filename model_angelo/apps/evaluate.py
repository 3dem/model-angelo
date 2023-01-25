"""
ModelAngelo evaluate model
Compare a predicted model with a ground truth model. You need:
1) A predicted mmCIF file, passed to --predicted-structure/--p/-p
2) A target mmCIF file, passed to --target-structure/--t/-t
"""
import numpy as np
import torch
from Bio.SVDSuperimposer import SVDSuperimposer
from model_angelo.utils.save_pdb_utils import chain_atom14_to_cif

from model_angelo.utils.cas_utils import get_correspondence, get_lddt
from model_angelo.utils.misc_utils import setup_logger
from model_angelo.utils.protein import Protein, get_protein_from_file_path
from model_angelo.utils.residue_constants import atom_order, atomc_backbone_mask


def get_all_atom_fit_report(
    input_protein: Protein,
    target_protein: Protein,
    max_dist=5,
    verbose=False,
    two_rounds=False,
    output_structure=None,
):
    input_cas = np.zeros_like(input_protein.atom_positions[:, 0])
    target_cas = np.zeros_like(target_protein.atom_positions[:, 0])
    # Protein parts
    input_cas[input_protein.prot_mask] = input_protein.atom_positions[
        input_protein.prot_mask, atom_order["CA"]
    ]
    input_cas[~input_protein.prot_mask] = input_protein.atom_positions[
        ~input_protein.prot_mask, atom_order["P"]
    ]
    target_cas[target_protein.prot_mask] = target_protein.atom_positions[
        target_protein.prot_mask, atom_order["CA"]
    ]
    target_cas[~target_protein.prot_mask] = target_protein.atom_positions[
        ~target_protein.prot_mask, atom_order["P"]
    ]

    target_correspondence, input_correspondence = get_correspondence(
        input_cas, target_cas, max_dist, verbose, two_rounds=two_rounds
    )

    if len(target_correspondence) == 0:
        return 0, 0, 0, 0, 0

    false_positive_count = len(
        set(range(len(input_cas))).difference(input_correspondence)
    )
    false_negative_count = len(
        set(range(len(target_cas))).difference(target_correspondence)
    )
    true_positive_count = len(input_correspondence)

    input_cas_cor = input_cas[input_correspondence]
    target_cas_cor = target_cas[target_correspondence]

    input_atoms = input_protein.atomc_positions[input_correspondence]
    target_atoms = target_protein.atomc_positions[target_correspondence]
    input_mask = input_protein.atomc_mask[input_correspondence]
    target_mask = target_protein.atomc_mask[target_correspondence]

    sup = SVDSuperimposer()
    sup.set(target_cas_cor, input_cas_cor)
    sup.run()
    rot, trans = sup.get_rotran()

    input_atoms_superimposed = np.einsum("nad,db->nab", input_atoms, rot) + trans[None]

    distance = np.linalg.norm(target_atoms - input_atoms_superimposed, axis=-1)
    backbone_rms = np.sum(
        input_mask * target_mask * atomc_backbone_mask * distance
    ) / np.sum(input_mask * atomc_backbone_mask * target_mask)
    ca_rms = np.sum((input_mask * target_mask * distance)[..., 1]) / np.sum(
        (input_mask * target_mask)[..., 1]
    )

    lddt_score = get_lddt(torch.Tensor(input_cas_cor), torch.Tensor(target_cas_cor))
    sequence_match = np.sum(
        input_protein.aatype[input_correspondence]
        == target_protein.aatype[target_correspondence]
    ) / len(target_correspondence)

    if output_structure is not None:
        new_bfactors = np.zeros_like(input_protein.b_factors[:, 0])
        correct_idxs = (
            input_protein.aatype[input_correspondence]
            == target_protein.aatype[target_correspondence]
        ).astype(np.float32)
        new_bfactors[input_correspondence] = 100 * correct_idxs
        chain_atom14_to_cif(
            [input_protein.aatype[c] for c in input_protein.chain_idx_to_residues],
            [
                input_protein.atomc_positions[c]
                for c in input_protein.chain_idx_to_residues
            ],
            [input_protein.atomc_mask[c] for c in input_protein.chain_idx_to_residues],
            path_to_save=output_structure,
            bfactors=[new_bfactors[c] for c in input_protein.chain_idx_to_residues],
        )

    return (
        backbone_rms,
        ca_rms,
        lddt_score.mean(),
        true_positive_count / (true_positive_count + false_negative_count),
        true_positive_count / (true_positive_count + false_positive_count),
        sequence_match,
    )


def add_args(parser):
    parser.add_argument(
        "--predicted-structure",
        "--p",
        "-p",
        required=True,
        help="Where your prediction cif file is",
    )
    parser.add_argument(
        "--target-structure",
        "--t",
        "-t",
        required=True,
        help="Where your real model cif file is",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=3,
        help="In angstrom (Å), the maximum distance for correspondence",
    )
    parser.add_argument("--output-file", help="If set, saves the results to a file")
    parser.add_argument(
        "--output-structure",
        help="If set, saves the sequence recall results to an mmCIF file, "
        "B-factors of 100 correspond to correct classifications and "
        "B-factors of 0 correspond to wrong classifications",
    )
    parser.add_argument(
        "--csv-format",
        action="store_true",
        help="If set, writes results in comma separated format.",
    )
    parser.add_argument(
        "--name", type=str, default="", help="Name of structure, to add to the csv"
    )
    return parser


def main(parsed_args):
    if parsed_args.output_file is not None:
        dir_name = os.path.dirname(parsed_args.output_file)
        os.makedirs(dir_name, exist_ok=True)
        logger = setup_logger(parsed_args.output_file)
        print_fn = logger.info
    else:
        print_fn = print

    predicted_protein = get_protein_from_file_path(parsed_args.predicted_structure)
    target_protein = get_protein_from_file_path(parsed_args.target_structure)
    (
        rmsd,
        ca_rms,
        lddt_score,
        recall,
        precision,
        sequence_match,
    ) = get_all_atom_fit_report(
        predicted_protein,
        target_protein,
        max_dist=parsed_args.max_dist,
        verbose=False,
        two_rounds=True,
        output_structure=parsed_args.output_structure,
    )

    if parsed_args.output_file is not None:
        with open(os.path.join(dir_name, "evaluation.pkl"), "wb") as p_wf:
            pickle.dump(
                {
                    "rmsd": rmsd,
                    "recall": recall,
                    "precision": precision,
                    "sequence_match": sequence_match,
                },
                p_wf,
            )

    if not parsed_args.csv_format:
        print_fn("*" * 50)
        print_fn(
            f"Results for \nPrediction file: {parsed_args.predicted_structure}\n"
            f"Target file: {parsed_args.target_structure}\n"
            f"Maximum distance of {parsed_args.max_dist} Å are"
        )
        print_fn("*" * 50)

        print_fn(
            f"**** Backbone RMSD:      {rmsd:.3f} Å\n"
            f"**** Cα RMSD:            {ca_rms:.3f} Å\n"
            f"**** Recall:             {recall:.3f}\n"
            f"**** Precision:          {precision:.3f}\n"
            f"**** lDDT score:         {lddt_score:.3f}\n"
            f"**** Sequence match:     {sequence_match:.3f}\n"
            f"**** Sequence coverage:  {sequence_match * recall:.3f}"
        )
    else:
        print(
            f"{parsed_args.name},"
            f"{rmsd:.3f},"
            f"{ca_rms:.3f},"
            f"{100*recall:.1f},"
            f"{100*precision:.1f},"
            f"{100*lddt_score:.1f},"
            f"{100*sequence_match:.1f},"
            f"{100 * sequence_match * recall:.1f}"
        )


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
