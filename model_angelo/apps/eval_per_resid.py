"""
ModelAngelo evaluate model residue per residue
Compare a predicted model with a ground truth model. You need:
1) A predicted mmCIF file, passed to --predicted-structure/--p/-p
2) A target mmCIF file, passed to --target-structure/--t/-t
3) Output file, passed to --output-file/--o/-o

or
1) An input data file
"""
import argparse
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

from model_angelo.utils.cas_utils import get_correspondence
from model_angelo.utils.protein import Protein, get_protein_from_file_path
from model_angelo.utils.residue_constants import atom_order, atomc_backbone_mask
from scipy.stats.mstats import mquantiles


def get_residue_fit_report(
    input_protein: Protein,
    target_protein: Protein,
    max_dist=5,
    verbose=False,
    two_rounds=False,
    eps=1e-9,
):
    input_cas = input_protein.atom_positions[:, atom_order["CA"]]
    target_cas = target_protein.atom_positions[:, atom_order["CA"]]
    target_correspondence, input_correspondence, unmatched_target_idxs, unmatched_input_idxs = get_correspondence(
        input_cas, target_cas, max_dist, verbose, two_rounds=two_rounds, get_unmatched=True
    )

    target_correspondence, input_correspondence, unmatched_target_idxs, unmatched_input_idxs = (
        target_correspondence.astype(np.int32),
        input_correspondence.astype(np.int32),
        unmatched_target_idxs.astype(np.int32),
        unmatched_input_idxs.astype(np.int32),
    )

    input_cas_cor = input_cas[input_correspondence]
    target_cas_cor = target_cas[target_correspondence]

    input_atoms = input_protein.atom_positions[input_correspondence]
    target_atoms = target_protein.atom_positions[target_correspondence]
    input_mask = input_protein.atom_mask[input_correspondence]
    target_mask = target_protein.atom_mask[target_correspondence]

    sup = SVDSuperimposer()
    sup.set(target_cas_cor, input_cas_cor)
    sup.run()
    rot, trans = sup.get_rotran()

    input_atoms_superimposed = np.einsum("nad,db->nab", input_atoms, rot) + trans[None]

    distance = np.linalg.norm(target_atoms - input_atoms_superimposed, axis=-1)

    input_type = np.concatenate(
        [
            input_protein.aatype[input_correspondence],
            np.full(len(unmatched_target_idxs), -1),
            input_protein.aatype[unmatched_input_idxs]
        ],
        0
    )

    target_type = np.concatenate(
        [
            target_protein.aatype[target_correspondence],
            target_protein.aatype[unmatched_target_idxs],
            np.full(len(unmatched_input_idxs), -1)
        ],
        0
    )

    input_residue_index = np.concatenate(
        [
            input_protein.residue_index[input_correspondence],
            np.full(len(unmatched_target_idxs), -1),
            input_protein.residue_index[unmatched_input_idxs]
        ],
        0
    )

    target_residue_index = np.concatenate(
        [
            target_protein.residue_index[target_correspondence],
            target_protein.residue_index[unmatched_target_idxs],
            np.full(len(unmatched_input_idxs), -1)
        ],
        0
    )

    ca_rms = (input_mask * target_mask * distance)[..., 1]
    ca_rms = np.concatenate(
        [
            ca_rms,
            np.full(len(input_type) - len(ca_rms), -1)
        ],
        0
    )

    backbone_rms = np.sum(
        input_mask * target_mask * atomc_backbone_mask * distance
        , 
        -1
    ) / (
            np.sum(input_mask * atomc_backbone_mask * target_mask, -1) + eps
    )

    

    backbone_rms = np.concatenate(
        [
            backbone_rms,
            np.full(len(input_type) - len(backbone_rms), -1)
        ],
        0
    )

    unmatched_target_mask = target_protein.atom_mask[unmatched_target_idxs]
    b_factors = np.concatenate(
        [
            np.sum(target_protein.b_factors[target_correspondence] * target_mask, -1) / np.sum(target_mask, -1),
            np.sum(target_protein.b_factors[unmatched_target_idxs] * unmatched_target_mask, -1) / np.sum(unmatched_target_mask, -1),
            np.full(len(unmatched_input_idxs), -1)
        ],
        0
    )

    matching_aa_mask = input_protein.aatype[input_correspondence] == target_protein.aatype[target_correspondence]
    all_atom_rms = (
            np.sum((input_mask * target_mask * distance), axis=-1) / np.sum(input_mask * target_mask, axis=-1)
    )[matching_aa_mask]
    all_atom_rms = np.concatenate(
        [
            all_atom_rms,
            np.full(len(b_factors) - len(all_atom_rms), -1)
        ],
        0
    )

    return input_type, target_type, input_residue_index, target_residue_index, ca_rms, backbone_rms, all_atom_rms, b_factors


def plot_data(input_type, target_type, ca_rms, backbone_rms, all_atom_rms, b_factors):
    bin_edges = mquantiles(b_factors[b_factors >= 0], prob=np.linspace(0., 1., 40))
    bin_centers = []

    target_mask = target_type >= 0  # Were there is a target

    input_type_masked = input_type[target_mask]
    target_type_masked = target_type[target_mask]
    ca_rms_masked = ca_rms[target_mask]
    backbone_rms = backbone_rms[target_mask]
    all_atom_rms = all_atom_rms[target_mask]
    b_factors_masked = b_factors[target_mask]

    mean_seq_recall = []
    mean_ca_recall = []
    mean_ca_rms = []
    mean_backbone_rms = []
    mean_all_atom_rms = []
    count = []
    for i in range(len(bin_edges) - 1):
        bfac_mask = (bin_edges[i] <= b_factors_masked) & (b_factors_masked < bin_edges[i + 1])
        c = np.sum(bfac_mask)

        if c < 2:
            continue

        bin_centers.append(bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2)
        count.append(c)

        mean_seq_recall.append(
            np.sum(input_type_masked[bfac_mask] == target_type_masked[bfac_mask]) / (np.sum(bfac_mask) + 1e-9)
        )

        bfac_match_mask = (input_type_masked >= 0) & bfac_mask
        mean_ca_recall.append(np.sum(bfac_match_mask) / (np.sum(bfac_mask) + 1e-9))
        mean_ca_rms.append(np.mean(ca_rms_masked[bfac_match_mask]))
        mean_backbone_rms.append(np.mean(backbone_rms[bfac_match_mask]))
        mean_all_atom_rms.append(np.mean(all_atom_rms[bfac_match_mask]))

    import matplotlib.pylab as plt
    f = plt.figure(1)
    plt.bar(bin_centers, count)

    f = plt.figure(2)
    plt.plot(bin_centers, mean_seq_recall)
    plt.title("Sequence recall")
    plt.ylim([0., 1.])

    f = plt.figure(3)
    plt.plot(bin_centers, mean_ca_recall)
    plt.title("C-alpha recall")
    plt.ylim([0., 1.])

    f = plt.figure(4)
    plt.plot(bin_centers, mean_ca_rms)
    plt.title("C-alpha RMS")
    plt.ylim([0., 2.2])

    f = plt.figure(5)
    plt.plot(bin_centers, mean_backbone_rms)
    plt.title("Backbone RMS")
    plt.ylim([0., 2.2])

    f = plt.figure(6)
    plt.plot(bin_centers, mean_all_atom_rms)
    plt.title("All Atom RMS")
    plt.ylim([0., 1.])

    plt.show()


def add_args(parser):
    parser.add_argument(
        "--predicted-structure",
        "--p",
        "-p",
        help="Where your prediction cif file is",
    )
    parser.add_argument(
        "--target-structure",
        "--t",
        "-t",
        help="Where your real model cif file is",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=3,
        help="In angstrom (Å), the maximum distance for correspondence",
    )
    parser.add_argument(
        "--output-file",
        "--o",
        "-o",
        help="Saves the results to table file with this path"
    )
    parser.add_argument(
        "--data-file",
        "--d",
        "-d",
        help="Reads from this data file"
    )
    parser.add_argument(
        "--plot",
        help="Plot results output figures",
        action="store_true"
    )
    parser.add_argument(
        "--skip-header",
        help="Don't print header in output file",
        action="store_true"
    )
    return parser


def main(parsed_args):
    if parsed_args.data_file is not None:
        db = np.loadtxt(parsed_args.data_file, dtype=str, delimiter="\t", skiprows=1)
        input_type = db[:, 0].astype(int)
        target_type = db[:, 1].astype(int)
        input_residue_index = db[:, 2].astype(int)
        target_residue_index = db[:, 3].astype(int)
        ca_rms = db[:, 4].astype(float)
        backbone_rms = db[:, 5].astype(float)
        all_atom_rms = db[:, 6].astype(float)
        b_factors = db[:, 7].astype(float)
    else:
        predicted_protein = get_protein_from_file_path(parsed_args.predicted_structure)
        target_protein = get_protein_from_file_path(parsed_args.target_structure)

        (
            input_type,
            target_type,
            input_residue_index,
            target_residue_index,
            ca_rms,
            backbone_rms,
            all_atom_rms,
            b_factors
        ) = get_residue_fit_report(
            predicted_protein,
            target_protein,
            max_dist=parsed_args.max_dist,
            verbose=False,
            two_rounds=True,
        )

    if parsed_args.plot:
        plot_data(input_type, target_type, ca_rms, backbone_rms, all_atom_rms, b_factors)

    if parsed_args.output_file is not None and parsed_args.data_file is None:
        with open(parsed_args.output_file, 'w') as f:
            if not parsed_args.skip_header:
                f.write(
                    "input_type\t"
                    "target_type\t"
                    "input_residue_index\t"
                    "target_residue_index\t"
                    "ca_rms\t"
                    "backbone_rms\t"
                    "all_atom_rms\t"
                    "b_factors\n"
                )
            for i in range(len(input_type)):
                f.write(
                    f"{input_type[i]}\t"
                    f"{target_type[i]}\t"
                    f"{input_residue_index[i]}\t"
                    f"{target_residue_index[i]}\t"
                    f"{ca_rms[i]}\t"
                    f"{backbone_rms[i]}\t"
                    f"{all_atom_rms[i]}\t"
                    f"{b_factors[i]}\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
