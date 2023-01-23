import os
import sys

import numpy as np
import torch
import tqdm
from loguru import logger

from model_angelo.gnn.flood_fill import final_results_to_cif
from model_angelo.utils.gnn_inference_utils import init_empty_collate_results, get_inference_data, collate_nn_results, \
    argmin_random, run_inference_on_data, init_protein_from_see_alpha, get_neighbour_idxs, get_final_nn_results, \
    get_base_parser
from model_angelo.utils.grid import MRCObject, make_model_angelo_grid, load_mrc
from model_angelo.utils.misc_utils import abort_if_relion_abort
from model_angelo.utils.protein import (
    get_protein_from_file_path,
    load_protein_from_prot,
)
from model_angelo.utils.torch_utils import (
    checkpoint_load_latest,
    get_model_from_file,
)


def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_angelo_output_dir = os.path.dirname(args.output_dir)

    module = get_model_from_file(os.path.join(args.model_dir, "model.py"))
    step = checkpoint_load_latest(
        args.model_dir,
        torch.device("cpu"),
        match_model=False,
        model=module,
    )
    logger.info(f"Loaded module from step: {step}")

    module = module.eval().to(args.device)

    if hasattr(module, "voxel_size"):
        voxel_size = module.voxel_size
    else:
        voxel_size = 1.5

    protein = None
    if args.struct.endswith("prot"):
        protein = load_protein_from_prot(args.struct)
    elif args.struct.endswith("cif") or args.struct.endswith("pdb"):
        if "output" in args.struct:
            protein = init_protein_from_see_alpha(args.struct)
        else:
            protein = get_protein_from_file_path(args.struct)
    if protein is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")

    grid_data = None
    if args.map.endswith("mrc"):
        grid_data = load_mrc(args.map, multiply_global_origin=False)
        grid_data = make_model_angelo_grid(
            grid_data.grid,
            grid_data.voxel_size,
            grid_data.global_origin,
            target_voxel_size=voxel_size,
        )
        grid_data = MRCObject(
            grid=grid_data.grid,
            voxel_size=grid_data.voxel_size,
            global_origin=np.zeros((3,), dtype=np.float32),
        )
    if grid_data is None:
        raise RuntimeError(
            f"Grid volume file {args.map} is not a supported file format."
        )

    num_res = len(protein.rigidgroups_gt_frames)

    collated_results = init_empty_collate_results(
        num_res,
        device="cpu",
    )

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)

    # Get an initial set of pointers to neighbours for more efficient inference
    init_neighbours = get_neighbour_idxs(protein, k=args.crop_length // 4)
    
    while residues_left > 0:
        idxs = argmin_random(collated_results["counts"], init_neighbours, args.batch_size)
        data = get_inference_data(protein, grid_data, idxs, crop_length=args.crop_length)
        results = run_inference_on_data(module, data, fp16=args.fp16)
        for i in range(args.batch_size):
            collated_results, protein = collate_nn_results(
                collated_results,
                results,
                data["indices"],
                protein,
                offset=i * args.crop_length,
            )
        residues_left = (
            num_res
            - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
        )
        steps_left = (
            total_steps
            - torch.sum(
                collated_results["counts"].clip(0, args.repeat_per_residue)
            ).item()
        )

        pbar.update(n=int(steps_left_last - steps_left))
        steps_left_last = steps_left
        abort_if_relion_abort(model_angelo_output_dir)

    pbar.close()

    final_results = get_final_nn_results(collated_results)
    output_path = os.path.join(args.output_dir, "output.cif")

    # Aggressive pruning does not make sense here
    final_results_to_cif(
        final_results=final_results,
        protein=protein,
        cif_path=output_path,
        verbose=True,
        print_fn=logger.info,
        aggressive_pruning=False,
        refine=args.refine,
    )

    return output_path


if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()
    infer(args)
