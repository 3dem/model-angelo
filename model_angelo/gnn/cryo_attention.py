import math

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from model_angelo.gnn.backbone_distance_embedding import BackboneDistanceEmbedding
from model_angelo.models.bottleneck import Bottleneck
from model_angelo.models.common_modules import FcResBlock, SpatialAvg, LayerNormNoBias
from model_angelo.utils.affine_utils import get_affine_rot
from model_angelo.utils.grid import (
    sample_centered_cube_rot_matrix,
    sample_centered_rectangle_along_vector,
)
from model_angelo.utils.residue_constants import canonical_num_residues
from model_angelo.utils.torch_utils import get_batches_to_idx


class CryoAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        neighbour_vector_embedding_dim: int = 64,
        attention_heads: int = 8,
        query_points: int = 4,
        num_neighbours: int = 20,
        q_length: int = 5,
        p_context: int = 17,
        cryo_emb_dim: int = 256,
        activation_class: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        assert cryo_emb_dim % 4 == 0

        self.ifz = in_features
        self.nvz = neighbour_vector_embedding_dim
        self.ahz = attention_heads
        self.qpz = query_points
        self.kz = num_neighbours

        self.attention_scale = math.sqrt(self.ifz)

        self.backbone_distance_emb = BackboneDistanceEmbedding(
            num_neighbours=num_neighbours,
            positional_encoding_dim=20,
            neighbour_vector_embedding_dim=neighbour_vector_embedding_dim,
            freq_inv=50,
        )

        self.ag = nn.Sequential(
            nn.Linear(self.ahz * self.ifz * 2, self.ifz, bias=False), nn.Dropout(p=0.5),
        )
        self.en = LayerNormNoBias(self.ifz)

        # Cryo stuff
        self.q_length = q_length
        self.p_context = p_context
        self.cryo_emb_dim = cryo_emb_dim

        self.cryo_point_v = nn.Sequential(
            Bottleneck(1, self.cryo_emb_dim // 4, stride=2, affine=True),
            Bottleneck(
                self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True
            ),
            Bottleneck(
                self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True
            ),
            Bottleneck(
                self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True
            ),
            SpatialAvg(),
            nn.Linear(self.cryo_emb_dim, self.ahz * self.ifz, bias=False),
            nn.Dropout(p=0.1),
        )

        self.cryo_vectors_k = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.cryo_emb_dim,
                kernel_size=(1, 3, 3),
                bias=False,
            ),
            Rearrange(
                "(b kz) c z y x -> b kz (c z y x)",
                kz=self.kz,
                c=self.cryo_emb_dim,
                z=self.q_length,
                x=1,
                y=1,
            ),
            LayerNormNoBias(self.cryo_emb_dim * self.q_length),
            activation_class(),
            nn.Linear(
                self.cryo_emb_dim * self.q_length, self.ahz * self.ifz, bias=False
            ),
            nn.Dropout(p=0.1),
            Rearrange(
                "b kz (ahz ifz) -> b kz ahz ifz",
                kz=self.kz,
                ahz=self.ahz,
                ifz=self.ifz,
            ),
        )
        self.cryo_q = nn.Sequential(
            nn.Linear(self.ifz + self.nvz, self.ahz * self.ifz, bias=False),
            Rearrange("n (ahz ifz) -> n ahz ifz", ahz=self.ahz, ifz=self.ifz,),
        )
        self.cryo_v = nn.Sequential(
            nn.Linear(self.ifz + self.nvz, self.ahz * self.ifz, bias=False),
            Rearrange("n (ahz ifz) -> n ahz ifz", ahz=self.ahz, ifz=self.ifz,),
        )

        self.cryo_edge_prediction_head = nn.Sequential(
            FcResBlock(
                self.ahz * (self.ifz + 1) + 23,
                self.ifz,
                activation_class=activation_class,
            ),
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            nn.Linear(self.ifz, 1),
        )

        self.cryo_point_aa_head = nn.Sequential(
            FcResBlock(
                self.ahz * self.ifz, self.ifz, activation_class=activation_class
            ),
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            nn.Linear(self.ifz, canonical_num_residues),
        )

    def forward(
        self,
        x,
        affines,
        prot_mask,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        edge_index=None,
        batch=None,
        **kwargs,
    ):
        """Calculates point wise attention to each query point and updates features

        Args:
            x: Node features of shape (N, _if)
            affines: Affine matrices describing positions of the nodes, of shape (N, 3, 4)
            cryo_grids: List of grids of length batch_size
            cryo_global_origins: List of global origins of grids
            cryo_voxel_sizes: List of voxel sizes of grids
            edge_index: Graph connectivity, of shape (N, k)
            batch: If using Pytorch Geometric graph batching, this is crucial
        """
        assert cryo_grids is not None
        dtype = x.dtype

        bde_out = self.backbone_distance_emb(x, affines, prot_mask, edge_index, batch)

        # Cryo-EM embedding and lookup
        batch_to_idx = (
            get_batches_to_idx(batch)
            if batch is not None
            else [torch.arange(0, len(x), dtype=int, device=x.device)]
        )
        with torch.no_grad():
            batch_cryo_grids = [
                cg.expand(len(b) * self.kz, -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_vectors = bde_out.neighbour_positions.detach()
            cryo_vectors = [cryo_vectors[b].reshape(-1, 3) for b in batch_to_idx]
            cryo_vectors_center_positions = [
                (
                    bde_out.positions[b]
                    .unsqueeze(1)
                    .expand(len(b), self.kz, 3)
                    .reshape(-1, 3)
                    - go
                )
                / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]
            cryo_vectors_rec = sample_centered_rectangle_along_vector(
                batch_cryo_grids,
                cryo_vectors,
                cryo_vectors_center_positions,
                rectangle_length=self.q_length,
            )  # (N kz) self.q_length 3 3
        cryo_vectors_query = self.cryo_q(bde_out.x_ne)  # N ahz ifz
        cryo_vectors_key = self.cryo_vectors_k(
            cryo_vectors_rec.requires_grad_()
        )  # N kz ahz ifz
        cryo_vectors_value = self.cryo_v(bde_out.x_ne)[
            bde_out.edge_index
        ]  # N kz ahz ifz
        cryo_vectors_attention_scores = (
            torch.einsum("nai,nkai->nka", cryo_vectors_query, cryo_vectors_key)
            / self.attention_scale
        )
        attention_weights = torch.softmax(
            cryo_vectors_attention_scores, dim=1,  # N kz ahz
        ).to(dtype)
        new_features_cryo_vectors = torch.einsum(
            "nkai,nka->nai", cryo_vectors_value, attention_weights
        )
        new_features_cryo_vectors = einops.rearrange(
            new_features_cryo_vectors,
            "n ahz ifz -> n (ahz ifz)",
            ahz=self.ahz,
            ifz=self.ifz,
        )

        cryo_edge_prediction_features = torch.cat(
            (
                cryo_vectors_value.flatten(2),  # N kz (ahz * ifz)
                bde_out.neighbour_positions,  # N kz 3
                bde_out.neighbour_distances,  # N kz 20
                attention_weights.detach(),  # N kz ahz
            ),
            dim=-1,
        )
        cryo_edge_prediction_logits = self.cryo_edge_prediction_head(
            cryo_edge_prediction_features
        ).reshape(-1, self.kz)

        with torch.no_grad():
            batch_cryo_grids = [
                cg.expand(len(b), -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_points = [
                (bde_out.positions[b].reshape(-1, 3) - go) / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]

            cryo_points_rot_matrices = [
                get_affine_rot(affines[b]).reshape(-1, 3, 3) for b in batch_to_idx
            ]

            cryo_points_cube = sample_centered_cube_rot_matrix(
                batch_cryo_grids,
                cryo_points_rot_matrices,
                cryo_points,
                cube_side=self.p_context,
            )
        new_features_cryo_points = self.cryo_point_v(cryo_points_cube.requires_grad_())
        cryo_aa_logits = self.cryo_point_aa_head(new_features_cryo_points)
        new_features_attention = torch.cat(
            (new_features_cryo_vectors, new_features_cryo_points,), dim=-1,
        )  # N (ahz * ifz) * 2
        new_features = self.ag(new_features_attention)  # Back to (N, ifz)
        new_features = self.en(x + new_features / math.sqrt(2)).to(dtype)
        return (
            new_features,
            bde_out.full_edge_index,
            cryo_edge_prediction_logits,
            cryo_aa_logits,
        )
