import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SpatialAvg(nn.Module):
    def forward(self, x):
        return x.mean(dim=[-3, -2, -1])


class SpatialMax(nn.Module):
    def forward(self, x):
        return x.view(*x.shape[:2], -1).max(dim=-1)[0]


class NormalizeStd(nn.Module):
    def __init__(self, first_channel_only=False) -> None:
        super().__init__()
        self.forward = (
            self.first_channel_only_forward
            if first_channel_only
            else self.normal_forward
        )

    def first_channel_only_forward(self, x):
        dims = list(range(1, len(x.shape)))
        return torch.cat(
            (
                x[:, 0].unsqueeze(1) / (torch.std(x, dim=dims, keepdim=True) + 1e-6),
                x[:, 1:],
            ),
            dim=1,
        )

    def normal_forward(self, x):
        dims = list(range(1, len(x.shape)))
        return x / (torch.std(x, dim=dims, keepdim=True) + 1e-6)


class Normalize(nn.Module):
    def __init__(self, first_channel_only=False) -> None:
        super().__init__()
        self.forward = (
            self.first_channel_only_forward
            if first_channel_only
            else self.normal_forward
        )

    def first_channel_only_forward(self, x):
        dims = list(range(1, len(x.shape)))
        x_fc = x[:, 0].unsqueeze(1)
        mu = torch.mean(x_fc, dim=dims, keepdim=True)
        sigma = torch.std(x_fc, dim=dims, keepdim=True)
        return torch.cat(((x_fc - mu) / (sigma + 1e-6), x[:, 1:],), dim=1,)

    def normal_forward(self, x):
        dims = list(range(1, len(x.shape)))
        mu = torch.mean(x, dim=dims, keepdim=True)
        sigma = torch.std(x, dim=dims, keepdim=True)
        return (x - mu) / (sigma + 1e-6)


class MultiScaleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        kernel_sizes=[3, 5, 7],
        paddings=[1, 2, 3],
        activation_class=nn.ReLU,
        normalization_class=nn.InstanceNorm3d,
        conv_class=nn.Conv3d,
        final_conv_class=nn.Conv3d,
        **kwargs,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.convs = nn.ModuleDict(
            {
                f"conv_{i}_{kernel}": conv_class(
                    in_channels,
                    mid_channels,
                    kernel_size=kernel,
                    padding=padding,
                    **kwargs,
                )
                for (i, (kernel, padding)) in enumerate(zip(kernel_sizes, paddings))
            }
        )

        self.bn = normalization_class(mid_channels * len(kernel_sizes))
        self.act = activation_class()

        self.final_conv = final_conv_class(
            mid_channels * len(kernel_sizes),
            out_channels,
            kernel_size=1,
            padding=0,
            **kwargs,
        )

    def forward(self, x):
        out = torch.cat([self.convs[k](x) for k in self.convs], dim=1)
        out = self.act(out)
        out = self.bn(out)
        return self.final_conv(out)


class FcResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation_class=nn.ReLU,
        normalization_class=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, out_features, bias=False),)
        self.forward = (
            self.residual_forward
            if in_features == out_features
            else self.non_residual_forward
        )
        self.activation = nn.Sequential(
            activation_class(), normalization_class(out_features),
        )

    def residual_forward(self, x):
        y = self.net(x)
        return self.activation(x + y / math.sqrt(2))

    def non_residual_forward(self, x):
        return self.activation(self.net(x))


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim, None)
        self.max_positions = num_embeddings

    def forward(self, position: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if (position > self.max_positions).any():
            raise ValueError(
                f"Position index {position.max()} above maximum "
                f" sequence length of {self.max_positions}"
            )
        positions = position.long()
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=10000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x


class LearnedGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("gate", nn.Parameter(torch.zeros(1)))

    def forward(self, x, y):
        s = torch.sigmoid(self.gate)
        return s * x + (1 - s) * y
