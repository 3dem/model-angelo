import math
from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from model_angelo.models.common_modules import FcResBlock
from model_angelo.utils.residue_constants import canonical_num_residues
from model_angelo.utils.torch_utils import get_batch_slices, padded_sequence_softmax

SequenceAttentionOutput = namedtuple(
    "SequenceAttentionOutput", ["x", "seq_aa_logits", "seq_attention_scores"]
)


def get_batched_sequence_attention_scores(
    sequence_query,  # naf
    sequence_key,  # 1saf
    batch,  # n
    attention_scale,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_query.shape[0], *sequence_key.shape[1:3], device=device
    )  # nsa
    n_len, s_len = output.shape[:2]
    seq_batches = get_batch_slices(s_len, batch_size)
    sequence_query = sequence_query[:, None]
    for seq_batch in seq_batches:
        output[:, seq_batch] = (sequence_query * sequence_key[:, seq_batch][batch]).sum(
            dim=-1
        ) / attention_scale
    return output


def get_batched_sequence_attention_features(
    sequence_attention_weights,  # nsa
    sequence_value,  # 1saf
    batch,  # n
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_attention_weights.shape[0], *sequence_value.shape[2:], device=device
    )  # naf
    n_len, s_len = sequence_attention_weights.shape[:2]
    seq_batches = get_batch_slices(s_len, batch_size)

    for seq_batch in seq_batches:
        output += (
            sequence_attention_weights[:, seq_batch][..., None]
            * sequence_value[:, seq_batch][batch]
        ).sum(dim=1)
    return output  # naf


class SequenceAttention(nn.Module):
    def __init__(
        self,
        sequence_features: int,
        in_features: int,
        attention_features: int = None,
        attention_heads: int = 8,
        activation_class: nn.Module = nn.ReLU,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.sfz = sequence_features
        self.ifz = in_features
        self.afz = in_features if attention_features is None else attention_features
        self.ahz = attention_heads
        self.attention_scale = math.sqrt(self.afz)
        self.resid_scale = math.sqrt(2)

        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange(
                "n (ahz afz) -> n ahz afz",
                ahz=self.ahz,
                afz=self.afz,
            ),
        )
        self.k = nn.Sequential(
            nn.Linear(self.sfz, self.ahz * self.afz, bias=False),
            Rearrange(
                "b s (ahz afz) -> b s ahz afz",
                ahz=self.ahz,
                afz=self.afz,
            ),
        )
        self.v = nn.Sequential(
            nn.Linear(self.sfz, self.ahz * self.afz, bias=False),
            Rearrange(
                "b s (ahz afz) -> b s ahz afz",
                ahz=self.ahz,
                afz=self.afz,
            ),
        )

        self.ag = nn.Sequential(
            OrderedDict(
                [
                    (
                        "rearrange",
                        Rearrange(
                            "n ahz afz -> n (ahz afz)",
                            ahz=self.ahz,
                            afz=self.afz,
                        ),
                    ),
                    ("ln", nn.LayerNorm(self.ahz * self.afz)),
                    ("linear", nn.Linear(self.ahz * self.afz, self.ifz, bias=False)),
                    (
                        "dropout",
                        nn.Dropout(p=0.5),
                    ),
                ]
            )
        )
        self.seq_aa_head = nn.Sequential(
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            FcResBlock(self.ifz, self.ifz, activation_class=activation_class),
            nn.Linear(self.ifz, canonical_num_residues),
        )
        self.en = nn.LayerNorm(self.ifz)

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(
        self,
        x,
        packed_sequence_emb,
        packed_sequence_mask,
        prot_mask,
        batch=None,
        attention_batch_size=200,
        **kwargs
    ):
        return self._intern_forward(
            x, packed_sequence_emb, packed_sequence_mask, prot_mask, batch, attention_batch_size
        )

    def forward_checkpoint(
        self,
        x: torch.Tensor,
        packed_sequence_emb: torch.Tensor,
        packed_sequence_mask: torch.Tensor,
        prot_mask: torch.LongTensor,
        batch=None,
        attention_batch_size: int = 200,
        **kwargs,
    ) -> SequenceAttentionOutput:
        return torch.utils.checkpoint.checkpoint(
            self._intern_forward,
            x,
            packed_sequence_emb,
            packed_sequence_mask,
            prot_mask,
            batch,
            attention_batch_size,
            preserve_rng_state=False,
        )

    def _intern_forward(
            self,
            x: torch.Tensor,
            packed_sequence_emb: torch.Tensor,
            packed_sequence_mask: torch.Tensor,
            prot_mask: torch.LongTensor,
            batch,
            attention_batch_size: int,
    ) -> SequenceAttentionOutput:
        device = x.device
        dtype = x.dtype
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        batch = batch[prot_mask]

        sequence_query = self.q(x[prot_mask])  # (n, ahz, afz)
        sequence_key = self.k(packed_sequence_emb)  # (1, seq_len, ahz, afz)
        sequence_value = self.v(packed_sequence_emb)  # (1, seq_len, ahz, afz)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        ).to(dtype)
        batched_mask = packed_sequence_mask[batch].unsqueeze(-1)  # (n, seq_len, 1)
        # Since sequence emb was padded, do not consider the padded parts for attention
        sequence_attention_weights = padded_sequence_softmax(
            sequence_attention_scores, batched_mask, dim=1
        )

        new_features = torch.zeros_like(x)
        unpacked_sequence_attention_scores = torch.zeros(
            x.shape[0], *sequence_attention_scores.shape[1:], device=device, dtype=dtype,
        )
        seq_aa_logits = torch.zeros(
            x.shape[0], canonical_num_residues, device=device, dtype=dtype,
        )
        unpacked_sequence_attention_scores[prot_mask] = sequence_attention_scores

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch,
            batch_size=attention_batch_size,
            device=device,
        )
        new_features[prot_mask] = self.ag(new_features_attention).to(dtype)
        seq_aa_logits[prot_mask] = self.seq_aa_head(new_features[prot_mask])
        new_features = self.en(x + new_features / self.resid_scale).to(dtype)
        return SequenceAttentionOutput(
            x=new_features,
            seq_aa_logits=seq_aa_logits,
            seq_attention_scores=unpacked_sequence_attention_scores,
        )
