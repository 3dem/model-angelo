import os
import pickle
import sys
import time
from itertools import product
from random import shuffle

import numpy as np
import torch
import tqdm
from loguru import logger
from scipy.spatial import cKDTree

from model_angelo.data.dataset_preprocess import decompress_data
from model_angelo.utils.grid import (
    apply_bfactor_to_map,
    get_auto_mask,
    get_lattice_meshgrid_np,
    get_local_std,
    load_mrc,
    make_model_angelo_grid,
    save_mrc,
)
from model_angelo.utils.misc_utils import abort_if_relion_abort
from model_angelo.utils.save_pdb_utils import points_to_pdb, ca_ps_to_pdb
from model_angelo.utils.torch_utils import get_device_names
from model_angelo.models.multi_gpu_wrapper import MultiGPUWrapper



class NoCaAtomsError(Exception):
    pass



def grid_to_points(
    grid, threshold, neighbour_distance_threshold, prune_distance=1.1,
):
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape[-1], no_shift=True), -1)

    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]

    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0

        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p, prune_distance)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]

    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    return output_points, output_points_before_pruning


def backbone_trace_and_grid_to_points(
    backbone_trace,
    grid,
    backbone_trace_threshold,
    backbone_merge_distance_threshold,
    neighbour_distance_threshold,
):
    lattice = np.flip(
        get_lattice_meshgrid_np(backbone_trace.shape[-1], no_shift=True), -1
    )

    output_points_before_pruning = np.copy(
        lattice[backbone_trace > backbone_trace_threshold, :].reshape(-1, 3)
    )

    points = np.copy(output_points_before_pruning)
    probs = grid[backbone_trace > backbone_trace_threshold]

    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0

        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p, backbone_merge_distance_threshold)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum
            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]

    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    return output_points, output_points_before_pruning


def infer(args):
    crop = args.crop

    os.makedirs(args.output_path, exist_ok=True)
    model_angelo_output_dir = os.path.dirname(args.output_path)
    device_names = get_device_names(args.device)

    model_definition_path = os.path.join(args.log_dir, "model.py")
    state_dict_path = os.path.join(args.log_dir, args.model_checkpoint) 

    logger.info(f"Using model file {model_definition_path}")
    logger.info(f"Using checkpoint file {state_dict_path}")

    if args.map_path.endswith("pkl"):
        ds = pickle.load(open(args.map_path, "br"))
        ds["grid_io"] = decompress_data(ds["grid_io"])

        grid_np = ds["grid_io"]

        grid_np = np.copy(grid_np)
        global_origin = [0, 0, 0]
        voxel_size = ds["voxel_size"]

        if isinstance(ds["cas"], dict):
            cas = (ds["cas"]["CA"] - ds["global_origin"]) / ds["voxel_size"]
        else:
            cas = ds["cas"]

    elif args.map_path.endswith("mrc"):
        grid_np, voxel_size, global_origin = load_mrc(args.map_path)
        grid_np, voxel_size, global_origin = make_model_angelo_grid(
            np.copy(grid_np), voxel_size, global_origin, target_voxel_size=1.5
        )

    # grid_np = remove_dust_from_volume(grid_np)
    if args.bfactor != 0:
        grid_np = apply_bfactor_to_map(
            grid_np, voxel_size=voxel_size, bfactor=args.bfactor
        )

    if args.mask_path is not None:
        mask_data = load_mrc(args.mask_path)
        mask_grid, _, _ = make_model_angelo_grid(
            np.copy(mask_data.grid),
            mask_data.voxel_size,
            mask_data.global_origin,
            target_voxel_size=1.5,
        )
    elif args.auto_mask:
        mask_grid = get_auto_mask(grid_np, voxel_size)
    else:
        mask_grid = np.ones_like(grid_np)
    if not args.dont_mask_input:
        grid_np = grid_np * mask_grid
    grid_np = (grid_np - np.mean(grid_np)) / (np.std(grid_np) + 1e-6)

    grid = torch.Tensor(grid_np)
    output = torch.zeros(3, *grid.shape[-3:])
    count_output = torch.zeros(*grid.shape[-3:])
    grid_std = get_local_std(grid[None, None])[0, 0]
    grid_std /= torch.max(grid_std) + 1e-6

    bz = args.box_size
    stride = args.stride
    x_coordinates = [i * stride for i in range((grid.shape[-1] - bz) // stride)] + [
        grid.shape[-1] - 1 - bz
    ]
    y_coordinates = [i * stride for i in range((grid.shape[-1] - bz) // stride)] + [
        grid.shape[-1] - 1 - bz
    ]
    z_coordinates = [i * stride for i in range((grid.shape[-1] - bz) // stride)] + [
        grid.shape[-1] - 1 - bz
    ]
    coordinates_to_infer = list(product(x_coordinates, y_coordinates, z_coordinates))
    shuffle(coordinates_to_infer)

    i = 0

    logger.info(f"Input structure has shape: {grid_np.shape[-3:]}")
    logger.info("Running with these arguments:")
    logger.info(args)

    prediction_start_time = time.time()
    pbar = tqdm.tqdm(
        total=len(coordinates_to_infer), file=sys.stdout, position=0, leave=True,
    )
    with MultiGPUWrapper(model_definition_path, state_dict_path, device_names) as wrapper:
        while i < len(coordinates_to_infer):
            meta_batch_list = []
            meta_batch_coordinates = []
            for _ in device_names:
                batch_grid = []
                batch_coordinates = []
                for _ in range(args.batch_size):
                    if i < len(coordinates_to_infer):
                        curr_coordinate = coordinates_to_infer[i]
                        coordinate_slice = np.s_[
                            ...,
                            curr_coordinate[0] : curr_coordinate[0] + bz,
                            curr_coordinate[1] : curr_coordinate[1] + bz,
                            curr_coordinate[2] : curr_coordinate[2] + bz,
                        ]
                        sliced_grid = grid[coordinate_slice][None].clone()
                        sliced_grid_std = grid_std[coordinate_slice][None].clone()
                        batch_coordinates.append(curr_coordinate)
                        batch_grid.append(torch.cat((sliced_grid, sliced_grid_std), dim=0))
                        i += 1
                if len(batch_grid) > 0:
                    batch_grid = torch.stack(batch_grid)
                    meta_batch_list.append({"x": batch_grid})
                    meta_batch_coordinates.append(batch_coordinates)
            if len(meta_batch_list) > 0:
                meta_net_output = wrapper(meta_batch_list)
                pbar.update(sum(len(batch_grid["x"]) for batch_grid in meta_batch_list))
            abort_if_relion_abort(model_angelo_output_dir)

            for batch_coordinates, net_output in zip(meta_batch_coordinates, meta_net_output):
                for j, c in enumerate(batch_coordinates):
                    batch_slice = np.s_[
                        ...,
                        c[0] + crop : c[0] + bz - crop,
                        c[1] + crop : c[1] + bz - crop,
                        c[2] + crop : c[2] + bz - crop,
                    ]
                    net_output_batch = torch.sigmoid(
                        net_output[
                            j, ..., crop : bz - crop, crop : bz - crop, crop : bz - crop,
                        ]
                    )
                    output[batch_slice] += net_output_batch.cpu()
                    count_output[batch_slice] += 1

    pbar.close()

    prediction_end_time = time.time()
    prediction_time = prediction_end_time - prediction_start_time
    logger.info(
        f"Model prediction done, took {prediction_time:.2f} seconds for {len(coordinates_to_infer)} sliding windows"
    )
    logger.info(
        f"Average time is {1000 * prediction_time / len(coordinates_to_infer):.3f} ms"
    )

    output = output.cpu().float().numpy()
    count_output = count_output.cpu().float().numpy() + 1e-6
    output = output / count_output

    output = output * mask_grid[None]

    ca_grid = output[0]
    p_grid = output[1]
    backbone_trace = output[2]

    logger.info("Starting Cα grid to points...")
    output_ca_points, output_ca_points_before_pruning = grid_to_points(
        ca_grid, args.threshold, 6 / voxel_size
    )
    logger.info(
        f"Have {len(output_ca_points_before_pruning)} Cα points before pruning and {len(output_ca_points)} after pruning"
    )

    if len(output_ca_points) == 0:
        raise NoCaAtomsError(
            "No Cα atoms were predicted! This usually means the global resolution of \n"
            "the map is worse than 4 Å. It could also mean that the map is in the wrong hand."
        )
    
    points_to_pdb(
        os.path.join(args.output_path, "output_ca_points_before_pruning.cif"),
        voxel_size * output_ca_points_before_pruning,
    )
    output_file_path = os.path.join(args.output_path, "see_alpha_output_ca.cif")
    points_to_pdb(
        output_file_path, voxel_size * output_ca_points,
    )

    if args.do_nucleotides:
        logger.info("Starting P grid to points...")
        output_p_points, output_p_points_before_pruning = grid_to_points(
            p_grid, args.threshold, 10 / voxel_size, prune_distance=3.2 / voxel_size,
        )
        logger.info(
            f"Have {len(output_p_points_before_pruning)} P points before pruning and {len(output_p_points)} after pruning"
        )

        points_to_pdb(
            os.path.join(args.output_path, "output_p_points_before_pruning.cif"),
            voxel_size * output_p_points_before_pruning,
        )
        points_to_pdb(
            os.path.join(args.output_path, "see_alpha_output_p.cif"),
            voxel_size * output_p_points,
        )
        output_file_path = os.path.join(args.output_path, "see_alpha_merged_output.cif")
        ca_ps_to_pdb(
            output_file_path,
            voxel_size * output_ca_points,
            voxel_size * output_p_points,
        )

    if args.save_backbone_trace:
        logger.info("Saving backbone trace")
        save_mrc(
            backbone_trace.astype(np.float32),
            voxel_size,
            global_origin,
            os.path.join(args.output_path, "backbone_trace.mrc"),
        )

    if args.save_cryo_em_grid:
        save_mrc(
            grid_np.astype(np.float32),
            voxel_size,
            global_origin,
            os.path.join(args.output_path, "cryo_em_volume.mrc"),
        )
    if args.save_real_coordinates:
        points_to_pdb(
            os.path.join(args.output_path, "real_points.cif"), voxel_size * cas,
        )
    if args.save_output_grid:
        for i, name in enumerate(["ca_output_grid", "p_output_grid"]):
            save_mrc(
                output[i].astype(np.float32),
                voxel_size,
                global_origin,
                os.path.join(args.output_path, f"{name}.mrc"),
            )
    logger.info("Finished inference!")

    return output_file_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Where the model definition file and checkpoints are.",
    )
    parser.add_argument(
        "--model-checkpoint", required=True, help="Which model checkpoint file to use"
    )
    parser.add_argument(
        "--map-path", "--i", required=True, help="Where the map to predict is"
    )
    parser.add_argument(
        "--output-path",
        "--o",
        required=True,
        help="Where to save the output and log file",
    )
    parser.add_argument(
        "--mask-path", default=None, help="Solvent mask to apply to output",
    )
    parser.add_argument(
        "--bfactor", type=float, default=0, help="Bfactor to apply to the map",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to carry computations on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument(
        "--stride", type=int, default=32, help="The stride for inference"
    )
    parser.add_argument("--auto-mask", action="store_true", help="Generate auto mask.")
    parser.add_argument(
        "--dont-mask-input", action="store_true", help="Don't mask input grid."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Probability threshold for inference",
    )
    parser.add_argument(
        "--save-real-coordinates",
        action="store_true",
        help="Store true coordinates for easy comparison",
    )
    parser.add_argument(
        "--save-cryo-em-grid",
        action="store_true",
        help="Store the cryo-em grid that the network uses",
    )
    parser.add_argument(
        "--do-nucleotides",
        action="store_true",
        help="Predict nucleotide positions using P atom",
    )
    parser.add_argument(
        "--save-backbone-trace",
        action="store_true",
        help="Save predicted backbone trace of the grid",
    )
    parser.add_argument(
        "--save-output-grid",
        action="store_true",
        help="For debug purposes, the output grid of the network",
    )
    parser.add_argument(
        "--crop", type=int, default=6, help="Margin on each output window to discard"
    )
    parser.add_argument(
        "--box-size", type=int, default=64, help="Box size of CNN prediction"
    )
    args = parser.parse_args()

    infer(args,)
