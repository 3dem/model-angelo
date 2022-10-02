from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
from Bio.PDB import Chain, MMCIFParser, PDBParser, Residue
from Bio.PDB.Atom import DisorderedAtom
from Bio.PDB.Polypeptide import PPBuilder
from numpy.typing import ArrayLike

from model_angelo.utils.residue_constants import (
    backbone_atoms,
    index_to_nuc,
    restype3_to_atoms,
    restype_3to1,
    secondary_structure_to_simplified_index,
)

Helix = namedtuple(
    "Helix", ["chain", "start_res", "end_chain", "end_res", "helix_type"]
)
SheetStrand = namedtuple("SheetStrand", ["chain", "start_res", "end_chain", "end_res"])

SecondaryStructure = namedtuple(
    "SecondaryStructure", ["chain", "start_res", "end_res", "type"]
)
mmCIF_struct_conf = {
    "conf_type_id": 0,
    "id": 1,
    "pdbx_PDB_helix_id": 2,
    "beg_label_comp_id": 3,
    "beg_label_asym_id": 4,
    "beg_label_seq_id": 5,
    "pdbx_beg_PDB_ins_code": 6,
    "end_label_comp_id": 7,
    "end_label_asym_id": 8,
    "end_label_seq_id": 9,
    "pdbx_end_PDB_ins_code": 10,
    "beg_auth_comp_id": 11,
    "beg_auth_asym_id": 12,
    "beg_auth_seq_id": 13,
    "end_auth_comp_id": 14,
    "end_auth_asym_id": 15,
    "end_auth_seq_id": 16,
    "pdbx_PDB_helix_class": 17,
    "details": 18,
    "pdbx_PDB_helix_length": 19,
}


class CIFSecondaryStructures:
    def __init__(self, file_path=None) -> None:
        self.file_path = file_path
        self.helices = []
        self.sheets = []
        self.residue_to_type = {}

        if self.file_path is not None:
            self.parse_cif_file()

    def add_helix(
        self,
        start_chain: str,
        end_chain: str,
        start_res: int,
        end_res: int,
        helix_type: str,
    ) -> int:
        if start_chain != end_chain:
            return 1
        self.helices.append(
            Helix(
                chain=start_chain,
                start_res=start_res,
                end_chain=end_chain,  # Should be the same as chain, just for debugging
                end_res=end_res,
                helix_type=helix_type,
            )
        )

        if start_chain not in self.residue_to_type:
            self.residue_to_type[start_chain] = {}

        for residue in range(start_res, end_res + 1):
            self.residue_to_type[start_chain][residue] = f"HELIX_{helix_type}"

        return 0

    def add_sheet_strand(
        self, start_chain: str, end_chain: str, start_res: int, end_res: int
    ) -> int:
        if start_chain != end_chain:
            return 1
        self.sheets.append(
            SheetStrand(
                chain=start_chain,
                start_res=start_res,
                end_chain=end_chain,
                end_res=end_res,
            )
        )

        if start_chain not in self.residue_to_type:
            self.residue_to_type[start_chain] = {}

        for residue in range(start_res, end_res + 1):
            self.residue_to_type[start_chain][residue] = f"SHEET"

        return 0

    def parse_cif_file(self):
        with open(self.file_path, "r") as f:
            lines = f.readlines()
        will_see_helix = True
        will_see_sheet = True
        helix_is_next = False
        sheet_is_next = False
        for line in lines:
            if will_see_helix and line.startswith("HELX_P"):
                helix_is_next = True
                data = line.strip().split()
                self.add_helix(
                    start_chain=data[12],
                    end_chain=data[
                        15
                    ],  # Should be the same as chain, just for debugging
                    start_res=int(data[13]),
                    end_res=int(data[16]),
                    helix_type=data[-3],
                )
            if helix_is_next and line.startswith("#"):
                helix_is_next = False
                will_see_helix = False

            elif will_see_sheet and line.startswith(
                "_struct_sheet_range.end_auth_seq_id"
            ):
                sheet_is_next = True

            elif sheet_is_next:
                if line.startswith("#"):
                    will_see_sheet = False
                    sheet_is_next = False
                else:
                    data = line.strip().split()

                    self.add_sheet_strand(
                        start_chain=data[11],
                        end_chain=data[14],
                        start_res=int(data[12]),
                        end_res=int(data[15]),
                    )
            elif not will_see_sheet and not will_see_helix:
                break

    def get_residue_secondary_structure(self, chain_id: str, resseq: int) -> str:
        return self.residue_to_type.get(chain_id, {}).get(resseq, "NULL")


class DSSPSecondaryStructures:
    def __init__(self, file_path=None) -> None:
        self.file_path = file_path
        self.residue_to_type = {}
        self.structures = []

        if self.file_path is not None:
            self.parse_cif_file()

    def parse_cif_file(self):
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        struct_conf_begin = False
        for line in lines:
            if line.startswith("_struct_conf.pdbx_PDB_helix_length"):
                struct_conf_begin = True
            elif struct_conf_begin and line.startswith("#"):
                break
            elif struct_conf_begin:
                data = line.split()
                structure = SecondaryStructure(
                    chain=data[mmCIF_struct_conf["beg_auth_asym_id"]],
                    start_res=int(data[mmCIF_struct_conf["beg_auth_seq_id"]]),
                    end_res=int(data[mmCIF_struct_conf["end_auth_seq_id"]]),
                    type=data[mmCIF_struct_conf["conf_type_id"]],
                )
                self.structures.append(structure)

                if structure.chain not in self.residue_to_type:
                    self.residue_to_type[structure.chain] = {}

                for residue in range(structure.start_res, structure.end_res + 1):
                    self.residue_to_type[structure.chain][residue] = structure.type

    def get_residue_secondary_structure(self, chain_id: str, resseq: int) -> str:
        return self.residue_to_type.get(chain_id, {}).get(resseq, "OTHER")


def map_chain_to_global_residue(
    input_data: Union[Helix, SheetStrand], chain_lengths: List[int]
) -> Tuple[int, int]:
    assert input_data.chain.isalpha()
    chain_idx = ord(input_data.chain) - ord("A")
    offset = 0 if chain_idx == 0 else chain_lengths[chain_idx - 1]
    return input_data.start_res + offset, input_data.end_res + offset


def load_cas_from_structure(stu_fn, all_structs=False, quiet=True):
    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])

    structure = parser.get_structure("structure", stu_fn)
    if not quiet and len(structure) > 1:
        print(f"WARNING: {len(structure)} structures found in model file: {stu_fn}")

    if not all_structs:
        structure = [structure[0]]

    ca_coords = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")

        for i, a in enumerate(model.get_atoms()):
            if a.get_name() == "CA":
                if isinstance(a, DisorderedAtom):
                    ca_coords.append(
                        a.disordered_get_list()[0].get_vector().get_array()
                    )
                else:
                    ca_coords.append(a.get_vector().get_array())

    return np.array(ca_coords)


def load_cas_ps_from_structure(stu_fn, all_structs=False, quiet=True):
    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])

    structure = parser.get_structure("structure", stu_fn)
    if not quiet and len(structure) > 1:
        print(f"WARNING: {len(structure)} structures found in model file: {stu_fn}")

    if not all_structs:
        structure = [structure[0]]

    coords = {"CA": [], "P": []}
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")

        for i, a in enumerate(model.get_atoms()):
            aname = a.get_name()
            if aname == "CA" or aname == "P":
                if isinstance(a, DisorderedAtom):
                    coords[aname].append(
                        a.disordered_get_list()[0].get_vector().get_array()
                    )
                else:
                    coords[aname].append(a.get_vector().get_array())

    return {"CA": np.array(coords["CA"]), "P": np.array(coords["P"])}


def get_residue_resseq(residue: Residue) -> int:
    s = residue.__repr__().split()[3]
    return int(s.replace("resseq=", ""))


def get_chain_name(chain: Chain) -> str:
    c = chain.__repr__().replace("<", "")
    c = c.replace(">", "").replace("Chain id=", "")
    return c


def load_full_backbone_from_structure(
    stu_fn, all_models=False, quiet=True, dssp_secondary_structures=False
):
    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])

    models = parser.get_structure("structure", stu_fn)

    if dssp_secondary_structures:
        secondary_structures = DSSPSecondaryStructures(stu_fn)
    else:
        secondary_structures = CIFSecondaryStructures(stu_fn)

    if not quiet and len(models) > 1:
        print(f"WARNING: {len(models)} models found in model file: {stu_fn}")

    if not all_models:
        models = [models[0]]

    coords = {
        "CA": [],
        "C": [],
        "O": [],
        "N": [],
        "P": [],
    }
    residue_info = {
        "aa_type": [],
        "secondary_structure": [],
        "chain": [],
        "nuc_type": [],
    }

    for model in models:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for chain in model:
            chain_name = get_chain_name(chain)
            for r in chain.get_residues():
                residue_name = r.get_resname()
                if residue_name in restype_3to1:
                    resseq = get_residue_resseq(r)
                    residue_info["chain"].append(chain_name)
                    residue_info["secondary_structure"].append(
                        secondary_structures.get_residue_secondary_structure(
                            chain_name, resseq
                        )
                    )
                    residue_info["aa_type"].append(residue_name)
                    for a_name in ["CA", "C", "N", "O"]:
                        try:
                            a = r[a_name]
                            if isinstance(a, DisorderedAtom):
                                coords[a_name].append(
                                    a.disordered_get_list()[0].get_vector().get_array()
                                )
                            else:
                                coords[a_name].append(a.get_vector().get_array())
                        except:
                            # Missing atom
                            if not quiet:
                                print(f"Residue {r} missing atom {a_name}")
                            coords[a_name].append(np.array([np.nan, np.nan, np.nan]))
                elif residue_name in index_to_nuc:
                    resseq = get_residue_resseq(r)
                    residue_info["chain"].append(chain_name)
                    residue_info["secondary_structure"].append(
                        secondary_structures.get_residue_secondary_structure(
                            chain_name, resseq
                        )
                    )
                    residue_info["nuc_type"].append(residue_name)
                    try:
                        a = r["P"]
                        if isinstance(a, DisorderedAtom):
                            coords["P"].append(
                                a.disordered_get_list()[0].get_vector().get_array()
                            )
                        else:
                            coords["P"].append(a.get_vector().get_array())
                    except:
                        # Missing atom
                        if not quiet:
                            print(f"Residue {r} missing atom P")
                        coords["P"].append(np.array([np.nan, np.nan, np.nan]))

    coords = dict([(k, np.array(v)) for (k, v) in coords.items()])

    return coords, residue_info


AAResidue = namedtuple(
    "AAResidue",
    ["backbone_atoms", "side_chain_atoms", "secondary_structure", "aa_type", "chain"],
)
AtomGroup = namedtuple("AtomGroup", ["names", "coordinate_array"])


class ProteinStructure:
    def __init__(self, residues: List[AAResidue]) -> None:
        self.residues = residues

    def get_backbone_coords(self) -> Dict[str, ArrayLike]:
        coords = {"CA": [], "C": [], "N": [], "O": []}
        for residue in self.residues:
            for b_atom in coords:
                coords[b_atom].append(residue.backbone_atoms[b_atom])
        for b_atom in coords:
            coords[b_atom] = np.array(coords[b_atom])
        return coords

    def get_all_coords(self) -> ArrayLike:
        atoms = []
        for residue in self.residues:
            atoms.append(residue.backbone_atoms.coordinate_array)
            if residue.side_chain_atoms is not None:
                atoms.append(residue.side_chain_atoms.coordinate_array)
        return np.concatenate(atoms, axis=0)

    def get_all_secondary_structure(self, simplified: bool = False) -> List[str]:
        ss = []
        for residue in self.residues:
            ss.append(residue.secondary_structure)
        if simplified:
            return [secondary_structure_to_simplified_index[s] for s in ss]
        else:
            return ss

    def normalize_coords(self, global_origin, voxel_size) -> None:
        for i in range(len(self.residues)):
            self.residues[i].backbone_atoms.coordinate_array[:] -= global_origin
            self.residues[i].backbone_atoms.coordinate_array[:] /= voxel_size
            if self.residues[i].side_chain_atoms is not None:
                self.residues[i].side_chain_atoms.coordinate_array[:] -= global_origin
                self.residues[i].side_chain_atoms.coordinate_array[:] /= voxel_size

    def get_bound(self) -> Tuple[ArrayLike, ArrayLike]:
        min_coord = np.full([3], 1e6)
        max_coord = np.full([3], -1e6)
        for residue in self.residues:
            min_coord = np.min(
                [np.min(residue.backbone_atoms.coordinate_array, axis=0), min_coord],
                axis=0,
            )
            max_coord = np.max(
                [np.max(residue.backbone_atoms.coordinate_array, axis=0), max_coord],
                axis=0,
            )
            if residue.side_chain_atoms is not None:
                min_coord = np.min(
                    [
                        np.min(residue.side_chain_atoms.coordinate_array, axis=0),
                        min_coord,
                    ],
                    axis=0,
                )
                max_coord = np.max(
                    [
                        np.max(residue.side_chain_atoms.coordinate_array, axis=0),
                        max_coord,
                    ],
                    axis=0,
                )
        return (min_coord, max_coord)


def load_proteins_from_structure(
    stu_fn, all_models=False, quiet=True, dssp_secondary_structures=False
) -> ProteinStructure:
    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])

    models = parser.get_structure("structure", stu_fn)

    if dssp_secondary_structures:
        secondary_structures = DSSPSecondaryStructures(stu_fn)
    else:
        secondary_structures = CIFSecondaryStructures(stu_fn)

    if not quiet and len(models) > 1:
        print(f"WARNING: {len(models)} models found in model file: {stu_fn}")

    if not all_models:
        models = [models[0]]

    residues = []

    for model in models:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for chain in model:
            chain_name = get_chain_name(chain)
            for r in chain.get_residues():
                residue_name = r.get_resname()
                if residue_name in restype_3to1:
                    resseq = get_residue_resseq(r)
                    residue_atoms = {}
                    for a_name in restype3_to_atoms[residue_name]:
                        try:
                            a = r[a_name]
                            if isinstance(a, DisorderedAtom):
                                residue_atoms[a_name] = (
                                    a.disordered_get_list()[0].get_vector().get_array()
                                )

                            else:
                                residue_atoms[a_name] = a.get_vector().get_array()
                        except:
                            # Missing atom
                            if not quiet:
                                print(f"Residue {r} missing atom {a_name}")
                            residue_atoms[a_name] = np.array([np.nan, np.nan, np.nan])
                    residue_backbone = AtomGroup(
                        backbone_atoms,
                        np.array([residue_atoms[k] for k in backbone_atoms]),
                    )
                    residue_sidechain = None
                    if len(restype3_to_atoms[residue_name]) > 4:
                        residue_sidechain = AtomGroup(
                            names=restype3_to_atoms[residue_name][4:],
                            coordinate_array=np.array(
                                [
                                    residue_atoms[k]
                                    for k in restype3_to_atoms[residue_name][4:]
                                ]
                            ),
                        )
                    residues.append(
                        AAResidue(
                            backbone_atoms=residue_backbone,
                            side_chain_atoms=residue_sidechain,
                            secondary_structure=secondary_structures.get_residue_secondary_structure(
                                chain_name, resseq
                            ),
                            aa_type=residue_name,
                            chain=chain_name,
                        )
                    )
    return ProteinStructure(residues)


def get_distribution_of_ca_angles(cifs):
    angles = []
    parser = MMCIFParser(QUIET=True)
    ppb = PPBuilder()
    for cif in cifs:
        try:
            structure = parser.get_structure("st", cif)
            chains = []
            for pp in ppb.build_peptides(structure):
                chains.append(
                    np.array([c.get_vector().get_array() for c in pp.get_ca_list()])
                )
            for chain in chains:
                v = chain[1:] - chain[:-1]
                v /= np.linalg.norm(v, axis=-1, keepdims=True)
                chain_angles = np.arccos(np.dot(-v[:-1], v[1:].T))
                chain_angles = chain_angles[np.isfinite(chain_angles)]
                angles.append(chain_angles)
        except Exception as e:
            print(e)
    angles = np.concatenate(angles)
    return angles


def generate_sequence_and_chain(pdb_file):
    if pdb_file.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=True)
    elif pdb_file.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise RuntimeError("Unknown type for structure file:", pdb_file[-3:])

    models = parser.get_structure("structure", pdb_file)
    model = models[0]

    result = {
        "position": [],
        "aa_type": [],
    }

    for chain in model:
        result["aa_type"].append([])
        result["position"].append([])
        for r in chain.get_residues():
            residue_name = r.get_resname()
            if residue_name in restype_3to1:
                try:
                    a = r["CA"]
                    if isinstance(a, DisorderedAtom):
                        result["position"][-1].append(
                            a.disordered_get_list()[0].get_coord().tolist()
                        )
                    else:
                        result["position"][-1].append(a.get_coord().tolist())
                    result["aa_type"][-1].append(restype_3to1[residue_name])
                except:
                    # Missing atom
                    pass
    result["aa_type"] = ["".join(a) for a in result["aa_type"]]
    return result
