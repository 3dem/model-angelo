#!/usr/bin/python3

from __future__ import absolute_import, division, print_function

import argparse
import glob
import io
import os
import pickle
import sys

import numpy as np

from model_angelo.utils.fasta_utils import read_fasta
from model_angelo.utils.pdb_utils import (
    load_cas_from_structure,
    load_full_backbone_from_structure,
)

sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))

from model_angelo.utils.grid import (
    get_spectral_indices,
    load_mrc,
    make_model_angelo_grid,
    spectral_amplitude,
)


def compress_data(data):
    data_io = io.BytesIO()
    np.savez_compressed(data_io, data)
    return data_io


def decompress_data(data):
    data.seek(0)
    return np.load(data, allow_pickle=True)["arr_0"].astype(np.float32)


def main(
    mrc_fn,
    stu_fn,
    out,
    target_voxel_size,
    original_stu_fn=None,
    fasta_fn=None,
    do_backbone=False,
    fix_shift=True,
):

    ds = {}

    if "*" in mrc_fn:
        mrc_fn = glob.glob(mrc_fn)[0]

    grid, voxel_size, global_origin = load_mrc(mrc_fn)
    grid, voxel_size, global_origin = make_model_angelo_grid(
        grid, voxel_size, global_origin, target_voxel_size
    )

    print("Output voxel size:", voxel_size)

    shape = np.array(grid.shape)
    cella = shape * voxel_size

    print("Grid shape:", shape)

    grid_ft = np.fft.rfftn(grid)
    R = get_spectral_indices(grid_ft)
    grid_spectral_amplitude = spectral_amplitude(grid_ft, R)

    if do_backbone:
        atom_coords, residue_info = load_full_backbone_from_structure(stu_fn)
        ds["residue_info"] = residue_info
    else:
        atom_coords = load_cas_from_structure(stu_fn)

    if original_stu_fn:
        origin_shift = 0
        if do_backbone:
            original_atom_coords, residue_info = load_full_backbone_from_structure(
                original_stu_fn
            )
            ds[
                "residue_info"
            ] = residue_info  # Overwrite residue info from original cif file
            if fix_shift:
                origin_shift = np.mean(atom_coords["CA"], axis=0) - np.mean(
                    original_atom_coords["CA"], axis=0
                )

        else:
            if fix_shift:
                original_atom_coords = load_cas_from_structure(original_stu_fn)
                origin_shift = np.mean(atom_coords, axis=0) - np.mean(
                    original_atom_coords, axis=0
                )
        global_origin += origin_shift
        print("Shifted global origin by ", origin_shift)

    ds["global_origin"] = global_origin
    ds["cella"] = cella
    ds["voxel_size"] = voxel_size
    ds["shape"] = shape

    ds["grid_spectral_amplitude"] = grid_spectral_amplitude
    grid_io = compress_data(grid)
    ds["grid_io"] = grid_io
    ds["cas"] = atom_coords

    if fasta_fn is not None:
        ds["fasta"] = {}
        ds["fasta"]["sequences"] = read_fasta(fasta_fn, auth_chains=True)

        ds["fasta"]["chain_to_seqid"] = {}
        for seqid, fasta_sequence in enumerate(ds["fasta"]["sequences"]):
            for chain in fasta_sequence.chains:
                ds["fasta"]["chain_to_seqid"][chain] = seqid

    print("Writing file:", out)
    pickle.dump(ds, open(out, "wb"), protocol=4)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mrc",
        type=str,
        help="Multiple maps supported (wild card must be passed as string)",
    )
    parser.add_argument("stu", type=str, help="Structure file")
    parser.add_argument("out", type=str, help="Root filename")
    parser.add_argument(
        "--original_stu",
        type=str,
        default=None,
        help="If cleaned PDB, this is the original",
    )
    parser.add_argument(
        "--fasta-sequence",
        type=str,
        default=None,
        help="The optional FASTA sequence file",
    )
    parser.add_argument("--target_voxel_size", type=float, default=1.0)
    parser.add_argument(
        "--force", action="store_true", help="Force to redo the calculations"
    )
    parser.add_argument(
        "--do-backbone", action="store_true", help="Do the backbone instead of CAs"
    )
    parser.add_argument(
        "--dont-fix-shift",
        action="store_true",
        help="If using original stu, don't fix the shift",
    )
    args = parser.parse_args()

    print("Args:", args)
    if not os.path.isfile(args.out) or args.force:
        main(
            mrc_fn=args.mrc,
            stu_fn=args.stu,
            out=args.out,
            target_voxel_size=args.target_voxel_size,
            original_stu_fn=args.original_stu,
            fasta_fn=args.fasta_sequence,
            do_backbone=args.do_backbone,
            fix_shift=not args.dont_fix_shift,
        )
    else:
        print("Already done! Use --force option if you want to redo this file")
