import numpy as np
import torch

from model_angelo.gnn.violation_loss import (
    between_residue_bond_loss,
    kdtree_between_residue_clash_loss,
    within_residue_violations,
)
from model_angelo.utils.protein import get_protein_from_file_path
from model_angelo.utils.residue_constants import (
    atomc_dists_bounds,
    element_names_arr,
    van_der_waals_radius,
)
from model_angelo.utils.save_pdb_utils import atom14_to_cif
from model_angelo.utils.torch_utils import to_numpy


def relax(input_path, output_path, device="cpu", num_iterations: int = 500):
    protein = get_protein_from_file_path(input_path)
    atom14_pos = (
        torch.Tensor(protein.atom14_positions).to(device=device).requires_grad_()
    )
    atom14_mask = torch.Tensor(protein.atom14_mask).to(device=device)
    residue_index = torch.LongTensor(protein.residue_index).to(device=device)
    batch = torch.ones(atom14_pos.shape[0]).to(device=device)
    aatype = torch.LongTensor(protein.aatype).to(device=device)
    bfactor = torch.Tensor(protein.b_factors)[:, 0].to(device=device) / 100
    bfactor = 1 - bfactor

    atom14_dists_lower_bound = torch.from_numpy(
        atomc_dists_bounds["lower_bound"][to_numpy(aatype)]
    ).to(device=device)
    atom14_dists_upper_bound = torch.from_numpy(
        atomc_dists_bounds["upper_bound"][to_numpy(aatype)]
    ).to(device=device)
    atom14_atom_radius = torch.from_numpy(
        np.vectorize(lambda x: van_der_waals_radius[x])(
            element_names_arr[to_numpy(aatype)]
        )
    ).to(device=device)

    opt = torch.optim.LBFGS([atom14_pos], lr=0.5, max_iter=20)

    def loss_fn():
        bond_loss = (
            between_residue_bond_loss(
                atom14_pos, atom14_mask, residue_index, batch, aatype
            )["per_residue_bond_loss"]
            .mul(bfactor)
            .div(bfactor.sum())
            .sum()
        )
        residue_loss = (
            within_residue_violations(
                atom14_pred_positions=atom14_pos,
                atom14_atom_exists=atom14_mask,
                atom14_dists_lower_bound=atom14_dists_lower_bound,
                atom14_dists_upper_bound=atom14_dists_upper_bound,
            )["per_atom_residue_loss"]
            .sum(dim=-1)
            .mean()
        )
        clash_loss = (
            kdtree_between_residue_clash_loss(
                atom14_pos, atom14_mask, atom14_atom_radius, residue_index, batch,
            )["per_atom_clash_loss"]
            .sum(dim=-1)
            .mul(bfactor)
            .div(bfactor.sum())
            .sum()
        )

        loss = bond_loss + residue_loss + clash_loss
        return loss

    for i in range(num_iterations):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step(loss_fn)

        if i % 100 == 0:
            print(f"Loss at step {i} is {loss.item():.3f}")

    atom14_to_cif(
        protein.aatype,
        to_numpy(atom14_pos),
        protein.atom14_mask,
        output_path,
        protein.b_factors,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="Where the input model is")
    parser.add_argument(
        "--output-path", required=True, help="Where the output should be saved"
    )
    parser.add_argument("--device", default="cpu", help="What torch device to use")

    args = parser.parse_args()

    relax(args.input_path, args.output_path, device=args.device)
