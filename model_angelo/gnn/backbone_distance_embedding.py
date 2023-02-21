from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from model_angelo.utils.knn_graph import knn_graph

from model_angelo.models.common_modules import SinusoidalPositionalEncoding
from model_angelo.utils.affine_utils import get_affine_translation, vecs_to_local_affine
from model_angelo.utils.protein import frames_and_literature_positions_to_atom3_pos
from model_angelo.utils.residue_constants import num_prot

BackboneDistanceEmbeddingOutput = namedtuple(
    "BackboneDistanceEmbeddingOutput",
    [
        "x_ne",
        "positions",
        "neighbour_positions",
        "neighbour_distances",
        "edge_index",
        "full_edge_index",
    ],
)


class BackboneDistanceEmbedding(nn.Module):
    def __init__(
        self,
        num_neighbours: int = 20,
        positional_encoding_dim: int = 20,
        neighbour_vector_embedding_dim: int = 64,
        freq_inv: float = 50,
    ) -> None:
        super().__init__()
        self.kz = num_neighbours
        self.pd = positional_encoding_dim
        self.nvz = neighbour_vector_embedding_dim
        self.freq_inv = freq_inv

        self.distance_encoding = SinusoidalPositionalEncoding(
            self.pd, freq_inv=self.freq_inv
        )
        self.ne = nn.Linear(5 * self.kz * self.pd, self.nvz, bias=False)

    def forward(
        self, x, affines, prot_mask, edge_index=None, batch=None
    ) -> BackboneDistanceEmbeddingOutput:
        positions = get_affine_translation(affines)
        if edge_index is None:
            edge_index = knn_graph(
                positions, k=self.kz, batch=batch, loop=False, flow="source_to_target"
            )
            full_edge_index = edge_index
            edge_index = edge_index[0].reshape(len(positions), self.kz)
        neighbour_positions = vecs_to_local_affine(
            affines, positions[edge_index]
        )  # N kz 3
        neighbour_distances = self.distance_encoding(
            neighbour_positions.norm(dim=-1)
        )  # N kz pd
        pseudo_aatypes = np.zeros(len(affines), dtype=np.int64)
        pseudo_aatypes[(~prot_mask).cpu().numpy()] = num_prot
        ncac = frames_and_literature_positions_to_atom3_pos(
            pseudo_aatypes, affines
        )  # N 3 3

        ca_pos = ncac[:, 1][:, None][:, None]  # N 1 1 3

        n_pos = ncac[:, 0][:, None]  # N 1 3
        c_pos = ncac[:, 2][:, None]  # N 1 3

        ca_to_ncac_distances = (
            (ncac[edge_index] - ca_pos).norm(dim=-1).flatten(1)
        )  # N (kz 3)
        n_to_c_distances = (
            (ncac[edge_index][:, :, 2] - n_pos).norm(dim=-1).flatten(1)
        )  # N kz
        c_to_n_distances = (
            (ncac[edge_index][:, :, 0] - c_pos).norm(dim=-1).flatten(1)
        )  # N kz

        encoding_distances = self.distance_encoding(
            torch.cat(
                (ca_to_ncac_distances, n_to_c_distances, c_to_n_distances,), dim=-1,
            )
        ).flatten(
            1
        )  # N (kz 5 pd)

        neighbour_embedding = self.ne(encoding_distances)  # N nvz
        x_ne = torch.cat((x, neighbour_embedding), dim=-1)  # N (ifz + nvz)
        return BackboneDistanceEmbeddingOutput(
            x_ne=x_ne,
            positions=positions,
            neighbour_positions=neighbour_positions,
            neighbour_distances=neighbour_distances,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
        )
