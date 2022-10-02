import glob
import os

from model_angelo.utils.grid import (
    apply_bfactor_to_map,
    crop_center_along_z,
    get_auto_mask,
    load_mrc,
    make_model_angelo_grid,
    save_mrc,
)


def standardize_mrc(args):
    if not os.path.isfile(args.input_path):
        input_files = glob.glob(os.path.join(args.input_path, "*.mrc"))
    else:
        input_files = [args.input_path]
    structure_names = [
        os.path.split(file_name)[1].split(".")[0] for file_name in input_files
    ]

    if os.path.isfile(args.output_path):
        output_path = os.path.split(args.output_path)[0]
    else:
        output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)
    output_files = [
        os.path.join(output_path, f"{structure_name}_fixed.mrc")
        for structure_name in structure_names
    ]

    for input_file, output_file in zip(input_files, output_files):
        input_mrc = load_mrc(input_file, multiply_global_origin=False)

        new_grid = input_mrc.grid

        if args.bfactor_to_apply != 0:
            new_grid = apply_bfactor_to_map(
                new_grid, input_mrc.voxel_size, args.bfactor_to_apply
            )

        if args.crop_z > 0:
            new_grid = crop_center_along_z(new_grid, args.crop_z)

        corrected_mrc = make_model_angelo_grid(
            grid=new_grid,
            voxel_size=input_mrc.voxel_size,
            global_origin=input_mrc.global_origin,
            target_voxel_size=args.target_voxel_size,
        )

        new_grid = corrected_mrc.grid
        if args.auto_mask:
            mask = get_auto_mask(new_grid, corrected_mrc.voxel_size)
            new_grid = new_grid * mask

        save_mrc(
            new_grid, corrected_mrc.voxel_size, corrected_mrc.global_origin, output_file
        )

    return output_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        "--i",
        required=True,
        help="The input file(s) to be standardized",
    )
    parser.add_argument(
        "--output-path",
        "--o",
        required=True,
        help="The output file(s) to be standardized",
    )
    parser.add_argument(
        "--target-voxel-size",
        "--vz",
        type=float,
        default=1.5,
        help="The target voxel size",
    )
    parser.add_argument(
        "--crop-z",
        "--amyloid-cropping",
        type=int,
        default=0,
        help="If input is an amyloid, crop "
        "this many voxels from the center"
        " on the Z-axis",
    )
    parser.add_argument(
        "--auto-mask", action="store_true", help="Use auto-masking for the input files"
    )
    parser.add_argument(
        "--bfactor-to-apply",
        type=float,
        default=0,
        help="Apply bfactor to map. 0 does nothing.",
    )
    args = parser.parse_args()

    standardize_mrc(args)
