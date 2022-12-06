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
import glob

backbone_affine = []
torsion_angles = []
all_frames = []
all_atoms = []
orig_atoms = []
aatypes = []
for pdb in glob.glob("nuc/*cif"):
    prot = get_protein_from_file_path(pdb)
    m = ~prot.prot_mask
    aatypes.append(torch.from_numpy(prot.aatype[m]))
    backbone_affine.append(torch.from_numpy(prot.rigidgroups_gt_frames[:, 0][m]))
    torsion_angles.append(torch.from_numpy(prot.torsion_angles_sin_cos)[m])
    # all_frames_ = torsion_angles_to_frames(
    #     aatypes[-1],
    #     backbone_affine[-1],
    #     torsion_angles[-1],
    # )
    all_frames_ = torch.from_numpy(prot.rigidgroups_gt_frames)[m]
    all_frames.append(all_frames_)
    all_atoms.append(frames_and_literature_positions_to_atomc_pos(aatypes[-1], all_frames_))
    orig_atoms.append(torch.from_numpy(prot.atomc_positions[m]).float())
    # atom_mask = torch.from_numpy(prot.atomc_mask)
    # atom14_to_cif(
    #     prot.aatype,
    #     all_atoms,
    #     atom_mask,
    #     "test.cif",
    # )

backbone_affine = torch.cat(backbone_affine, dim=0)
torsion_angles = torch.cat(torsion_angles, dim=0)
all_frames = torch.cat(all_frames, dim=0)
all_atoms = torch.cat(all_atoms, dim=0)
orig_atoms = torch.cat(orig_atoms, dim=0)
aatypes = torch.cat(aatypes, dim=0)

from model_angelo.utils.affine_utils import get_affine_rot, get_affine_translation
import model_angelo.utils.residue_constants as rc
import torch


for aa in range(8):
    if torch.sum(aatypes == aa + 20).item() == 0:
        continue
    aa_str = rc.index_to_restype_3[aa + 20]
    print(aa_str)
    mask = aatypes == aa + 20
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
        atoms = orig_atoms[mask]
        R, t = get_affine_rot(frame6), get_affine_translation(frame6)
        x = torch.einsum("nxy,nbx->nby", R, atoms[:, atom_idx]-t[:,None]).mean(dim=0)
        for a, xa in zip(atom_names, x):
            print(a, xa)
