import torch
from model_angelo.utils.protein import (
    frames_and_literature_positions_to_atomc_pos,
    torsion_angles_to_frames,
    get_protein_from_file_path,
)
from model_angelo.utils.residue_constants import restype_atomc_mask, select_torsion_angles, restype3_to_atoms, num_prot
from model_angelo.utils.save_pdb_utils import (
    atom14_to_cif,
    write_chain_report, write_chain_probabilities,
)

prot = get_protein_from_file_path("6by7.cif")
backbone_affine = torch.from_numpy(prot.rigidgroups_gt_frames[:, 0])
torsion_angles = torch.from_numpy(prot.torsion_angles_sin_cos)
all_frames = torsion_angles_to_frames(
    prot.aatype,
    backbone_affine,
    torsion_angles,
)
# all_frames = torch.from_numpy(prot.rigidgroups_gt_frames)
all_atoms = frames_and_literature_positions_to_atomc_pos(prot.aatype, all_frames)
atom_mask = torch.from_numpy(prot.atomc_mask)
atom14_to_cif(
    prot.aatype,
    all_atoms,
    atom_mask,
    "test.cif",
)

from model_angelo.utils.affine_utils import get_affine_rot, get_affine_translation
import model_angelo.utils.residue_constants as rc
import torch
import numpy as np


for aa in range(8):
    if np.sum(prot.aatype == aa + 20) == 0:
        continue
    aa_str = rc.index_to_restype_3[aa + 20]
    print(aa_str)
    mask = prot.aatype == aa + 20
    for frame_idx in range(9):
        atoms = []
        atom_names = []
        for atom in rc.rigid_group_atom_positions[aa_str]:
            if atom[1] == frame_idx:
                atoms.append(rc.restype3_to_atoms_index[aa_str][atom[0]])
                atom_names.append(atom[0])
        atom_idx = torch.tensor(atoms, dtype=torch.long)
        # if aatype % 2 == 0:
        #     n_str = "N9"
        # else:
        #     n_str = "N1"
        # n_str = "O2'"
        frame6 = all_frames[mask, frame_idx]
        atoms = torch.from_numpy(prot.atomc_positions[mask]).float()
        R, t = get_affine_rot(frame6), get_affine_translation(frame6)
        x = torch.einsum("nxy,nbx->nby", R, atoms[:, atom_idx]-t[:,None]).mean(dim=0)
        for a, xa in zip(atom_names, x):
            print(a, xa)
