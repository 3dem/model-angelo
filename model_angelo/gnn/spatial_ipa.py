import math
from collections import OrderedDict

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from model_angelo.gnn.backbone_distance_embedding import BackboneDistanceEmbedding
from model_angelo.utils.affine_utils import affine_mul_vecs


class SpatialIPA(nn.Module):
    def __init__(
            self,
            in_features: int,
            neighbour_vector_embedding_dim: int = 64,
            attention_heads: int = 8,
            query_points: int = 4,
            num_neighbours: int = 20,
            checkpoint: bool = False,  # Note, checkpointing is buggy currently for this class
            **kwargs,
    ):
        super().__init__()
        self.ifz = in_features
        self.nvz = neighbour_vector_embedding_dim
        self.ahz = attention_heads
        self.qpz = query_points
        self.kz = num_neighbours

        self.backbone_distance_emb = BackboneDistanceEmbedding(
            num_neighbours=num_neighbours,
            positional_encoding_dim=20,
            neighbour_vector_embedding_dim=neighbour_vector_embedding_dim,
            freq_inv=50,
        )

        self.loc_q = nn.Sequential(
            nn.Linear(self.ifz + self.nvz, self.ahz * self.qpz * 3, bias=False),
            Rearrange(
                "n (kz ahz qpz c) -> n kz ahz qpz c",
                ahz=self.ahz,
                qpz=self.qpz,
                kz=1,
                c=3,
            ),
        )
        self.loc_v = nn.Sequential(
            nn.Linear(self.ifz + self.nvz, self.ahz * self.qpz * self.ifz, bias=False),
            Rearrange(
                "n (ahz qpz ifz) -> n ahz qpz ifz",
                ahz=self.ahz,
                qpz=self.qpz,
                ifz=self.ifz,
            ),
        )

        self.ag = nn.Sequential(
            OrderedDict(
                [
                    (
                        "rearrange",
                        Rearrange(
                            "n ahz qpz ifz -> n (ahz qpz ifz)",
                            ahz=self.ahz,
                            qpz=self.qpz,
                            ifz=self.ifz,
                        ),
                    ),
                    ("ln", nn.LayerNorm(self.ahz * self.ifz * self.qpz)),
                    (
                        "linear",
                        nn.Linear(self.ahz * self.ifz * self.qpz, self.ifz, bias=False),
                    ),
                    ("dropout", nn.Dropout(p=0.5)),
                ]
            )
        )
        self.en = nn.LayerNorm(self.ifz)

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x, affines, prot_mask, edge_index=None, batch=None, **kwargs):
        """Calculates point wise attention to each query point and updates features

        Args:
            x: Node features of shape (N, _if)
            affines: Affine matrices describing positions of the nodes, of shape (N, 3, 4)
            edge_index: Graph connectivity, of shape (N, k)
            batch: If using Pytorch Geometric graph batching, this is crucial
        """
        dtype = x.dtype
        bde_out = self.backbone_distance_emb(x, affines, prot_mask, edge_index, batch)

        loc_query = self.loc_q(
            bde_out.x_ne
        )  # This should be with respect to the local frame
        loc_query = affine_mul_vecs(affines, loc_query)  # Bring it to global
        loc_value = self.loc_v(bde_out.x_ne)[bde_out.edge_index]

        global_neighbour_positions = einops.rearrange(
            bde_out.positions[
                bde_out.edge_index
            ],  # The global location of the neighbour positions
            "n (kz ahz qpz) c -> n kz ahz qpz c",
            ahz=1,
            qpz=1,
            kz=self.kz,
        )

        # Closer points should have higher attention score
        loc_attention_scores = -torch.norm(
            loc_query - global_neighbour_positions, dim=-1, p=2  # N kz ahz qpz
        ).sum(
            dim=-1
        )  # N kz ahz

        attention_weights = torch.softmax(
            loc_attention_scores,
            dim=1,  # N kz ahz
        ).to(dtype)
        new_features_loc = torch.einsum("nkaqi,nka->naqi", loc_value, attention_weights)

        new_features = self.ag(new_features_loc).to(dtype)  # Back to (N, ifz)
        new_features = self.en(x + new_features / math.sqrt(2)).to(dtype)
        return new_features, bde_out.edge_index

    def forward_checkpoint(self, x, affines, prot_mask, edge_index=None, batch=None, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            self.forward_normal,
            x,
            affines,
            prot_mask,
            edge_index,
            batch,
            preserve_rng_state=False,
        )
