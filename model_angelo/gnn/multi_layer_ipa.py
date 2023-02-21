import contextlib

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from model_angelo.gnn.backbone_frame_net import BackboneFrameNet
from model_angelo.gnn.cryo_attention import CryoAttention
from model_angelo.gnn.gnn_output import GNNOutput
from model_angelo.gnn.sequence_attention import SequenceAttention
from model_angelo.gnn.spatial_ipa import SpatialIPA
from model_angelo.models.common_modules import FcResBlock
from model_angelo.utils.misc_utils import assertion_check
from model_angelo.utils.residue_constants import canonical_num_residues


class MultiLayerSeparableIPA(nn.Module):
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

        self.seq_attentions = nn.ModuleList(
            [
                SequenceAttention(
                    sequence_features=in_features,
                    in_features=hidden_features,
                    attention_heads=attention_heads,
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
            nn.Linear(self.hfz, (canonical_num_residues * 5 + 3) * 2, bias=True),
            Rearrange("n (f d) -> n f d", f=canonical_num_residues * 5 + 3, d=2,),
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
        self.sequence_cache = []

    def forward(
        self,
        sequence=None,
        sequence_mask=None,
        prot_mask=None,
        positions=None,
        init_affine=None,
        record_training: bool = False,
        run_iters: int = 1,
        seq_attention_batch_size: int = 200,
        using_cache: bool = False,
        **kwargs,
    ) -> GNNOutput:
        assertion_check(
            sequence is not None
            and sequence_mask is not None
            and positions is not None
            and prot_mask is not None
        )
        dtype = positions.dtype
        result = GNNOutput(
            positions=positions,
            prot_mask=prot_mask,
            init_affine=init_affine,
            hidden_features=self.hfz,
        )
        self.init_training_record()
        if not using_cache:
            self.sequence_cache = []
        for run_iter in range(run_iters):
            not_last_iter = run_iter != (run_iters - 1)
            with torch.no_grad() if not_last_iter else contextlib.nullcontext():
                for idx in range(self.num_layers):
                    (
                        result["x"],
                        cryo_edges,
                        cryo_edge_logits,
                        cryo_aa_logits,
                    ) = self.cryo_attentions[idx](
                        x=result["x"],
                        affines=result["pred_affines"][-1],
                        prot_mask=prot_mask,
                        **kwargs,
                    )
                    self.append_to_training_record(
                        result["x"],
                        f"x_cryo_attention_{idx}",
                        record_training=record_training,
                    )
                    if using_cache:
                        sequence_key_cache, sequence_value_cache = self.sequence_cache[idx]
                    else:
                        sequence_key_cache, sequence_value_cache = (None,None)
                    (
                        result["x"],
                        seq_aa_logits,
                        seq_attention_scores,
                        sequence_key_cache,
                        sequence_value_cache,
                    ) = self.seq_attentions[idx](
                        x=result["x"],
                        packed_sequence_emb=sequence,
                        packed_sequence_mask=sequence_mask,
                        attention_batch_size=seq_attention_batch_size,
                        prot_mask=prot_mask,
                        sequence_key_cache=sequence_key_cache,
                        sequence_value_cache=sequence_value_cache,
                        **kwargs,
                    )
                    if not using_cache:
                        self.sequence_cache.append(
                            (sequence_key_cache.contiguous(), sequence_value_cache.contiguous())
                        )
                    self.append_to_training_record(
                        result["x"],
                        f"x_seq_attentions_{idx}",
                        record_training=record_training,
                    )
                    result["x"], _ = self.spatial_ipas[idx](
                        x=result["x"],
                        affines=result["pred_affines"][-1],
                        prot_mask=prot_mask,
                        **kwargs,
                    )
                    self.append_to_training_record(
                        result["x"],
                        f"x_spatial_ipa_{idx}",
                        record_training=record_training,
                    )
                    # Transition
                    result["x"] = self.transition_layer(result["x"]).to(dtype)
                    self.append_to_training_record(
                        result["x"],
                        f"transition_layer_x_{idx}",
                        record_training=record_training,
                    )
                    # Predict aa here
                    aa_contrib = (
                        torch.ones(
                            *cryo_aa_logits.shape[:-1],
                            1,
                            device=cryo_aa_logits.device,
                            dtype=dtype,
                        )
                        + prot_mask.to(dtype)[..., None]
                    )
                    cryo_aa_logits = (cryo_aa_logits + seq_aa_logits) / aa_contrib
                    # Predict backbone and N,CA,C atoms
                    ncac, new_affine = self.backbone_update_fc(
                        result["x"], result["pred_affines"][-1]
                    )
                    # Predict confidence score
                    local_confidence_score = self.local_confidence_predictor(
                        result["x"]
                    )
                    # Predict existence mask
                    pred_existence_mask = self.existence_mask_predictor(result["x"])

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
            # Save the last seq_attention_scores that will be used for location of residue prediction
            result["seq_attention_scores"] = seq_attention_scores

            if not_last_iter:
                result = GNNOutput(
                    positions=positions,
                    prot_mask=prot_mask,
                    init_affine=result["pred_affines"][-1],
                    hidden_features=self.hfz,
                )
            # By the end of the iter, the sequence is in the cache anyway
            using_cache = True
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
