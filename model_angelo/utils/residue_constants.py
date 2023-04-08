"""
Vast majority of this file comes from 
http://github.com/deepmind/alphafold/blob/main/alphafold/common/residue_constants.py
"""
import functools
import os
from collections import namedtuple
from typing import List, Mapping, Tuple

import numpy as np
import torch
import einops

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

restype_3_to_index = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}

restype_1_to_index = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}

index_to_restype_1 = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

index_to_restype_3 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

index_to_hmm_restype_1 = sorted(index_to_restype_1)
hmm_restype_1_to_index = {hmm_restype: i for i, hmm_restype in enumerate(index_to_hmm_restype_1)}
restype_1_order_to_hmm = [restype_1_to_index[aa] for aa in index_to_hmm_restype_1]

index_to_nuc = ["A", "C", "G", "U", "DA", "DC", "DG", "DT"]
nuc_atom_types = [
    "C1'",
    "C2",
    "C2'",
    "C3'",
    "C4",
    "C4'",
    "C5",
    "C5'",
    "C6",
    "C7",
    "C8",
    "N1",
    "N2",
    "N3",
    "N4",
    "N6",
    "N7",
    "N9",
    "O2",
    "O2'",
    "O3'",
    "O4",
    "O4'",
    "O5'",
    "O6",
    "OP1",
    "OP2",
    "P",
]
nuc_order = {nuctype: i for i, nuctype in enumerate(index_to_nuc)}
nuc_atom_order = {nuc_atom: i for i, nuc_atom in enumerate(nuc_atom_types)}

nuc_backbone_atoms = ["C5'", "P", "O5'"]

nuc_angles_atoms = {
    "A": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N9", "C8"],
    ],
    "C": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N1", "C6"],
    ],
    "G": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N9", "C8"],
    ],
    "U": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N1", "C6"],
    ],
    "DA": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N9", "C8"],
    ],
    "DC": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N1", "C6"],
    ],
    "DG": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N9", "C8"],
    ],
    "DT": [
        ["O5'", "C5'", "C4'", "O4'"],
        ["O4'", "C1'", "N1", "C6"],
    ],
}


nuc_rigid_group_atom_positions = {
    "A": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1526, -0.6821, -0.5093)],
        ["OP2", 0, (0.7474, 0.4337, -0.0707)],
        ["C5'", 0, (-2.6532, -0.0002, 0.000)],
        ["O5'", 0, (-1.4415, 0.7417, 0.000)],
        ["C4'", 0, (-3.7519, 0.7362, 0.1427)],
    ],
    "C": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1525, -0.6829, -0.5094)],
        ["OP2", 0, (0.7473, 0.4350, -0.0717)],
        ["C5'", 0, (-2.6494, 0.0001, 0.000)],
        ["O5'", 0, (-1.4372, 0.7425, 0.000)],
        ["C4'", 0, (-3.7467, 0.7380, 0.1387)],
    ],
    "G": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1507, -0.6845, -0.5141)],
        ["OP2", 0, (0.7492, 0.4386, -0.0701)],
        ["C5'", 0, (-2.6470, 0.0002, 0.000)],
        ["O5'", 0, (-1.4352, 0.7420, 0.000)],
        ["C4'", 0, (-3.7447, 0.7391, 0.1411)],
    ],
    "U": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1521, -0.6834, -0.5111)],
        ["OP2", 0, (0.7481, 0.4355, -0.0705)],
        ["C5'", 0, (-2.6449, 0.0003, 0.000)],
        ["O5'", 0, (-1.4329, 0.7422, 0.000)],
        ["C4'", 0, (-3.7437, 0.7370, 0.1392)],
    ],
    "DA": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1557, -0.6800, -0.5014)],
        ["OP2", 0, (0.7447, 0.4254, -0.0725)],
        ["C5'", 0, (-2.6388, 0.000, 0.000)],
        ["O5'", 0, (-1.4267, 0.7422, 0.000)],
        ["C4'", 0, (-3.7317, 0.7328, 0.1493)],
    ],
    "DC": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1459, -0.7001, -0.5190)],
        ["OP2", 0, (0.7580, 0.4510, -0.0782)],
        ["C5'", 0, (-2.6694, 0.000, 0.000)],
        ["O5'", 0, (-1.4568, 0.7424, 0.000)],
        ["C4'", 0, (-3.7592, 0.7297, 0.1481)],
    ],
    "DG": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1536, -0.6847, -0.5162)],
        ["OP2", 0, (0.7520, 0.4395, -0.0670)],
        ["C5'", 0, (-2.6789, 0.000, 0.000)],
        ["O5'", 0, (-1.4665, 0.7424, 0.000)],
        ["C4'", 0, (-3.7706, 0.7330, 0.1483)],
    ],
    "DT": [
        ["P", 0, (0.000, 0.000, 0.000)],
        ["OP1", 0, (0.1545, -0.6833, -0.5118)],
        ["OP2", 0, (0.7489, 0.4353, -0.0690)],
        ["C5'", 0, (-2.6713, 0.000, 0.000)],
        ["O5'", 0, (-1.4590, 0.7422, 0.000)],
        ["C4'", 0, (-3.7707, 0.7253, 0.1576)],
    ],
}

nuctype_to_atoms = {
    "A": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "O4'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],
    "G": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "O4'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "C": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "O4'",
        "N1",
        "C6",
        "C5",
        "C4",
        "N4",
        "N3",
        "C2",
        "O2",
    ],
    "U": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "O4'",
        "N1",
        "C6",
        "C5",
        "C4",
        "O4",
        "N3",
        "C2",
        "O2",
    ],
    "DA": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "O4'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],
    "DG": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "O4'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "DC": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "O4'",
        "N1",
        "C6",
        "C5",
        "C4",
        "N4",
        "N3",
        "C2",
        "O2",
    ],
    "DT": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "O4'",
        "N1",
        "C6",
        "C5",
        "C4",
        "O4",
        "N3",
        "C2",
        "O2",
        "C7",
    ],
}


restype_order = {restype: i for i, restype in enumerate(index_to_restype_1)}
restype_num = len(index_to_restype_1)

unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = index_to_restype_1 + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

atom37_backbone_mask = np.zeros((1, 37), dtype=np.float32)
atom37_backbone_mask[
    :,
    [
        atom_order["N"],
        atom_order["CA"],
        atom_order["C"],
        atom_order["O"],
    ]
] = 1

cif_secondary_structure_to_index = {
    "NULL": 0,
    "HELIX_1": 1,
    "HELIX_5": 2,
    "SHEET": 3,
}

ca_to_n_distance_ang = 1.4556349
ca_to_c_distance_ang = 1.5235157
peptide_bond_length_ang = 1.3310018

c_to_ca_to_n_angle_rad = 1.9384360


def parse_sequence_string(sequence: str) -> List[int]:
    return [restype_1_to_index[s] for s in sequence]


def parse_index_list(index_list: List[int]) -> List[str]:
    return "".join(index_to_restype_1[i] for i in index_list)


def translate_restype_3_to_1(residue_list: List[str]) -> List[str]:
    return [restype_3to1[r] for r in residue_list]


restype3_to_atoms = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

restype3_to_atoms_index = dict(
    [
        (res, dict([(a, i) for (i, a) in enumerate(atoms)]))
        for (res, atoms) in restype3_to_atoms.items()
    ]
)
for residue in restype3_to_atoms_index:
    restype3_to_atoms_index[residue]["OXT"] = restype3_to_atoms_index[residue]["O"]

backbone_atoms = set(["CA", "C", "N"])

secondary_structure_to_simplified_index = {
    # CIF
    "NULL": 0,
    "HELIX_1": 1,
    "HELIX_5": 1,
    "SHEET": 2,
    # DSSP
    "OTHER": 0,
    "BEND": 0,
    "TURN_TY1_P": 0,
    "HELX_RH_AL_P": 1,
    "HELX_RH_3T_P": 1,
    "HELX_LH_PP_P": 1,
    "HELX_RH_PI_P": 1,
    "STRN": 2,
}

# Distance from one CA to next CA [trans configuration: omega = 180].
ca_ca = 3.80209737096

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}


# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
rigid_group_atom_positions = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}

restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

atom14_backbone_mask = np.zeros((1, 14), dtype=np.float32)
atom14_backbone_mask[:, :4] = 1

atom14_names_arr = np.array(list(restype_name_to_atom14_names.values()))
element_names_arr = np.array(
    [
        [x if len(x) == 0 else x[:1] for x in y]
        for y in restype_name_to_atom14_names.values()
    ]
)


def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.
    ex_normalized = ex / np.linalg.norm(ex)

    # make ey perpendicular to ex
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=np.int64)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=np.int64)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_atom3_rigid_group_positions = np.zeros([21, 3, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)


def _make_rigid_group_constants():
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(index_to_restype_1):
        resname = restype_1to3[restype_letter]
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            restype_atom37_mask[restype, atomtype] = 1
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            restype_atom14_mask[restype, atom14idx] = 1
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position

            if atomname in backbone_atoms:
                restype_atom3_rigid_group_positions[
                    restype, atom14idx, :
                ] = atom_position

    for restype, restype_letter in enumerate(index_to_restype_1):
        resname = restype_1to3[restype_letter]
        atom_positions = {
            name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]
        }

        # backbone to backbone is the identity transform
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # pre-omega-frame to backbone (currently dummy identity matrix)
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

        # phi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["N"],
        )
        restype_rigid_group_default_frame[restype, 2, :, :] = mat

        # psi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],
            translation=atom_positions["C"],
        )
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        # chi1-frame to backbone
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            restype_rigid_group_default_frame[restype, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
residue_atom_renaming_swaps = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
    "P": 1.8,
    "": 0.0,
}


def sequence_to_onehot(
    sequence: str, mapping, map_unknown_to_x: bool = False
) -> torch.LongTensor:
    """Maps the given sequence into a one-hot encoded matrix.
    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain
        amino acid 'X', an error will be thrown. If False, any amino acid not in
        the mapping will throw an error.
    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
      the sequence.
    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot_arr = torch.zeros(len(sequence), num_entries, dtype=torch.long)

    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping["X"])
            else:
                raise ValueError(f"Invalid character in the sequence: {aa_type}")
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.
    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in residue_constants.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in index_to_restype_1:
        residue_name = restype_1to3[residue_name]
        residue_chi_angles = chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.asarray(chi_atom_indices)


chi_atom_indices = get_chi_atom_indices()


def get_nuc_angles_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.
    Returns:
      A tensor of shape [residue_types=8, chis=3, atoms=4]. The residue types are
      in the order specified in residue_constants.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in index_to_nuc:
        residue_chi_angles = nuc_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([nuc_atom_order[atom] for atom in chi_angle])
        chi_atom_indices.append(atom_indices)

    return np.asarray(chi_atom_indices)


nuc_angles_atom_indices = get_nuc_angles_atom_indices()


Bond = namedtuple("Bond", ["atom1_name", "atom2_name", "length", "stddev"])
BondAngle = namedtuple(
    "BondAngle", ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"]
)


@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[
    Mapping[str, List[Bond]], Mapping[str, List[Bond]], Mapping[str, List[BondAngle]]
]:
    """Load stereo_chemical_props.txt into a nice structure.
    Load literature values for bond lengths and bond angles and translate
    bond angles into the length of the opposite edge of the triangle
    ("residue_virtual_bonds").
    Returns:
      residue_bonds: Dict that maps resname -> list of Bond tuples.
      residue_virtual_bonds: Dict that maps resname -> list of Bond tuples.
      residue_bond_angles: Dict that maps resname -> list of BondAngle tuples.
    """
    stereo_chemical_props_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stereo_chemical_props.txt"
    )
    with open(stereo_chemical_props_path, "rt") as f:
        stereo_chemical_props = f.read()
    lines_iter = iter(stereo_chemical_props.splitlines())
    # Load bond lengths.
    residue_bonds = {}
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, length, stddev = line.split()
        atom1, atom2 = bond.split("-")
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(Bond(atom1, atom2, float(length), float(stddev)))
    residue_bonds["UNK"] = []

    # Load bond angles.
    residue_bond_angles = {}
    next(lines_iter)  # Skip empty line.
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, angle_degree, stddev_degree = line.split()
        atom1, atom2, atom3 = bond.split("-")
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180.0 * np.pi,
                float(stddev_degree) / 180.0 * np.pi,
            )
        )
    residue_bond_angles["UNK"] = []

    def make_bond_key(atom1_name, atom2_name):
        """Unique key to lookup bonds."""
        return "-".join(sorted([atom1_name, atom2_name]))

    # Translate bond angles into distances ("virtual bonds").
    residue_virtual_bonds = {}
    for resname, bond_angles in residue_bond_angles.items():
        # Create a fast lookup dict for bond lengths.
        bond_cache = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

            # Compute distance between atom1 and atom3 using the law of cosines
            # c^2 = a^2 + b^2 - 2ab*cos(gamma).
            gamma = ba.angle_rad
            length = np.sqrt(
                bond1.length ** 2
                + bond2.length ** 2
                - 2 * bond1.length * bond2.length * np.cos(gamma)
            )

            # Propagation of uncertainty assuming uncorrelated errors.
            dl_outer = 0.5 / length
            dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2
                + (dl_db1 * bond1.stddev) ** 2
                + (dl_db2 * bond2.stddev) ** 2
            )
            residue_virtual_bonds[resname].append(
                Bond(ba.atom1_name, ba.atom3name, length, stddev)
            )

    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)


def get_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=15):
    """compute upper and lower bounds for bonds to assess violations."""
    restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(index_to_restype_1):
        resname = restype_1to3[restype_letter]
        atom_list = restype_name_to_atom14_names[resname]

        # create lower and upper bounds for clashes
        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if (not atom2_name) or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 1e10
                restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

        # overwrite lower and upper bounds for bonds and angles
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {
        "lower_bound": restype_atom14_bond_lower_bound,  # shape (21,14,14)
        "upper_bound": restype_atom14_bond_upper_bound,  # shape (21,14,14)
        "stddev": restype_atom14_bond_stddev,  # shape (21,14,14)
    }


atom14_dists_bounds = get_atom14_dists_bounds()

# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
nuctype_atom28_to_rigid_group = np.zeros([8, 28], dtype=np.int64)
nuctype_atom28_mask = np.zeros([8, 28], dtype=np.float32)
nuctype_atom28_rigid_group_positions = np.zeros([8, 28, 3], dtype=np.float32)
nuctype_rigid_group_default_frame = np.zeros([8, 3, 4, 4], dtype=np.float32)


def _make_nuc_rigid_group_constants():
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(index_to_nuc):
        for atomname, group_idx, atom_position in nuc_rigid_group_atom_positions[
            restype_letter
        ]:
            atomtype = atom_order[atomname]
            nuctype_atom28_to_rigid_group[restype, atomtype] = group_idx
            nuctype_atom28_mask[restype, atomtype] = 1
            nuctype_atom28_rigid_group_positions[restype, atomtype, :] = atom_position

    for restype, restype_letter in enumerate(index_to_restype_1):
        atom_positions = {
            name: np.array(pos)
            for name, _, pos in nuc_rigid_group_atom_positions[restype_letter]
        }

        # backbone to backbone is the identity transform
        nuctype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # chi1 and chi2-frame to backbone
        for chi_index in range(2):
            base_atom_names = nuc_angles_atoms[restype_letter][chi_index]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nuctype_rigid_group_default_frame[restype, chi_index + 1, :, :] = mat


# _make_nuc_rigid_group_constants()

sequence_gap_to_distance = {
    1: {"mean": 3.8381097571710776, "std": 0.8309388293675871},
    2: {"mean": 6.0165053763264105, "std": 1.2389327661223906},
    3: {"mean": 7.328896550842753, "std": 2.382786963520511},
    4: {"mean": 8.966959468100962, "std": 3.088907030136974},
    5: {"mean": 10.914465646799231, "std": 3.410132853557923},
    6: {"mean": 12.410693726367242, "std": 3.9447030170701365},
    7: {"mean": 13.603296974171021, "std": 4.5485790427806005},
    8: {"mean": 14.90803959503693, "std": 4.97840744983613},
    9: {"mean": 16.15019000877649, "std": 5.378154149681046},
    10: {"mean": 17.131941321526323, "std": 5.819763512569835},
    11: {"mean": 18.009799834138207, "std": 6.24407068978648},
    12: {"mean": 18.895726269507502, "std": 6.635098989761409},
    13: {"mean": 19.67505298275992, "std": 7.0336528538926215},
    14: {"mean": 20.319752611207523, "std": 7.4308920002225465},
    15: {"mean": 20.92280694302011, "std": 7.832689065474961},
    16: {"mean": 21.50293563897015, "std": 8.2388425553948},
    17: {"mean": 22.003436761591377, "std": 8.635912794627034},
    18: {"mean": 22.440020886157534, "std": 9.02728225769117},
    19: {"mean": 22.86027894485405, "std": 9.419359409775721},
    20: {"mean": 23.254601335733966, "std": 9.802443051998265},
    21: {"mean": 23.603919343883035, "std": 10.161593188063058},
    22: {"mean": 23.927181328525794, "std": 10.50923792887111},
    23: {"mean": 24.239653833026647, "std": 10.853430370094836},
    24: {"mean": 24.531202519553613, "std": 11.181638818629349},
    25: {"mean": 24.7991202894309, "std": 11.493643815879782},
    26: {"mean": 25.055535719858987, "std": 11.802222294849251},
    27: {"mean": 25.3046184871037, "std": 12.098710899564287},
    28: {"mean": 25.54200314245827, "std": 12.37089809691424},
    29: {"mean": 25.77346088863421, "std": 12.622510312644746},
    30: {"mean": 26.00210154779639, "std": 12.863094714496427},
    31: {"mean": 26.222894891769954, "std": 13.090706028009434},
    32: {"mean": 26.434026776917158, "std": 13.302836670292479},
    33: {"mean": 26.638843084003575, "std": 13.50436175425381},
    34: {"mean": 26.839503489263105, "std": 13.696521622032273},
    35: {"mean": 27.037842564406663, "std": 13.869310832445779},
    36: {"mean": 27.23320880095485, "std": 14.026480431387146},
    37: {"mean": 27.425004858312573, "std": 14.176140300309632},
    38: {"mean": 27.61311570273596, "std": 14.318550294476236},
    39: {"mean": 27.799392016003125, "std": 14.449778186492944},
}


def select_torsion_angles(input, aatype):
    chi_angles = einops.rearrange(
        input[..., 3:, :], "... (f a) d -> ... f d a", f=4, a=20, d=2
    )[torch.arange(len(aatype)), ..., aatype]
    input_torsion_angles = torch.cat((input[..., :3, :], chi_angles), dim=-2)
    return input_torsion_angles
