import contextlib

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from model_angelo.models.common_modules import FcResBlock
from model_angelo.gnn.backbone_frame_net import BackboneFrameNet
from model_angelo.gnn.cryo_attention import CryoAttention
from model_angelo.gnn.spatial_ipa import SpatialIPA
from model_angelo.gnn.gnn_output import GNNOutput


class MultiLayerSeparableIPANoSeq(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            attention_heads: int = 8,
            query_points: int = 8,
            num_neighbours: int = 20,
            num_layers: int = 8,
            activation_class: nn.Module = nn.ReLU,
            **cryo_attention_kwargs,
    ) -> None:
        super().__init__()
        self.ifz = in_features
        self.hfz = hidden_features
        self.ahz = attention_heads
        self.qpz = query_points
        self.kz = num_neighbours
        self.num_layers = num_layers

        # Can't have affine since this sometimes has no gradients
        # self.input_ln = nn.LayerNorm(hidden_features, elementwise_affine=False)

        self.spatial_ipas = nn.ModuleList(
            [
                SpatialIPA(
                    in_features=hidden_features,
                    attention_heads=attention_heads,
                    query_points=query_points,
                    num_neighbours=num_neighbours,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.cryo_attentions = nn.ModuleList(
            [
                CryoAttention(
                    in_features=hidden_features,
                    attention_heads=attention_heads,
                    query_points=query_points,
                    num_neighbours=num_neighbours,
                    **cryo_attention_kwargs,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.transition_layer = nn.Sequential(
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
        )

        self.backbone_update_fc = BackboneFrameNet(
            self.hfz, activation_class=activation_class
        )

        # Predict the torsion angles ω, ϕ, ψ, χ_1, χ_2, χ_3, χ_4
        # The meaning of the χ's is dependent on the amino acid, so it is better to
        # predict 20 * 4 + 3 angles per residue
        self.torsion_angle_fc = nn.Sequential(
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            nn.Linear(self.hfz, (20 * 4 + 3) * 2, bias=True),
            Rearrange(
                "n (f d) -> n f d",
                f=83,
                d=2,
            ),
        )

        self.local_confidence_predictor = nn.Sequential(
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            nn.Linear(self.hfz, 1),
        )

        self.existence_mask_predictor = nn.Sequential(
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            nn.Linear(self.hfz, 1),
        )

        self.init_training_record()

    def forward(
            self,
            positions=None,
            init_affine=None,
            record_training=False,
            run_iters=1,
            **kwargs,
    ) -> GNNOutput:
        result = GNNOutput(positions=positions, init_affine=init_affine, hidden_features=self.hfz)
        self.init_training_record()

        for run_iter in range(run_iters):
            not_last_iter = run_iter != (run_iters - 1)
            with torch.no_grad() if not_last_iter else contextlib.nullcontext():
                for idx in range(self.num_layers):
                    # cryo_edge_probs is with respect to the current position's top neighbours
                    # Should calculate loss here
                    # You need cryo_edges so that you can index into the edge_exists matrix
                    (
                        result["x"],
                        cryo_edges,
                        cryo_edge_logits,
                        cryo_aa_logits,
                    ) = self.cryo_attentions[idx](
                        x=result["x"],
                        affines=result["pred_affines"][-1],
                        **kwargs,
                    )
                    self.append_to_training_record(
                        result["x"],
                        f"x_cryo_attention_{idx}",
                        record_training=record_training,
                    )
                    result["x"], _ = self.spatial_ipas[idx](
                        x=result["x"],
                        affines=result["pred_affines"][-1],
                        **kwargs,
                    )
                    self.append_to_training_record(
                        result["x"],
                        f"x_spatial_ipa_{idx}",
                        record_training=record_training,
                    )
                    # Transition
                    result["x"] = self.transition_layer(result["x"])
                    self.append_to_training_record(
                        result["x"],
                        f"transition_layer_x_{idx}",
                        record_training=record_training,
                    )
                    # Predict backbone and N,CA,C atoms
                    ncac, new_affine = self.backbone_update_fc(
                        result["x"], result["pred_affines"][-1]
                    )
                    # Predict confidence score
                    local_confidence_score = self.local_confidence_predictor(
                        result["x"]
                    )
                    # Predict existence mask
                    pred_existence_mask = self.existence_mask_predictor(
                        result["x"]
                    )

                    # Add data
                    result.update(
                        pred_ncac=ncac,
                        pred_affines=new_affine,
                        pred_positions=ncac[..., 1, :],
                        cryo_edges=cryo_edges,
                        cryo_edge_logits=cryo_edge_logits,
                        cryo_aa_logits=cryo_aa_logits,
                        local_confidence_score=local_confidence_score,
                        pred_existence_mask=pred_existence_mask,
                    )

            # Predict torsion angles as the 83 x 2 dimensional sin cos vectors
            result["pred_torsions"] = self.torsion_angle_fc(result["x"])

            if not_last_iter:
                result = GNNOutput(
                    positions=positions,
                    init_affine=result["pred_affines"][-1],
                    hidden_features=self.hfz,
                )
        return result

    @torch.no_grad()
    def append_to_training_record(self, tensor, name, record_training=False):
        if not record_training:
            return
        if name + "_grad_norm_mean" not in self._training_record:
            self._training_record[name + "_value_norm_mean"] = []
            self._training_record[name + "_value_norm_std"] = []
        dims = list(range(1, len(tensor.shape)))
        value_norm = tensor.norm(dim=dims)
        self._training_record[name + "_value_norm_mean"].append(
            value_norm.mean().item()
        )
        self._training_record[name + "_value_norm_std"].append(value_norm.std().item())

    def init_training_record(self):
        self._training_record = dict()

    @property
    def training_record(self):
        return dict([(k, np.mean(v)) for (k, v) in self._training_record.items()])
