import numpy as np
import torch
import torch.nn as nn

from model_angelo.models.common_modules import (
    AttentionBlock,
    GlobalLayer,
    MaxNeighbour,
    NormalizeStd,
    RegularBlock,
    ResBlock,
    Upsample
)


class Resnet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        g_width=12,
        num_blocks=10,
        activation_class=nn.ReLU,
        pooling_class=nn.MaxPool3d,
        normalization_class=nn.BatchNorm3d,
        end_activation_class=None,
        max_neighbour=False,
        long_connection=False,
        downsample_x=True,
        stochastic_depth_drop_factor=0,
        activation_block=False,
        patch_size=8,
        squeeze=False,
        global_layer=False,
    ) -> None:
        super().__init__()
        if activation_block:
            assert stochastic_depth_drop_factor == 0
        if global_layer:
            assert not activation_block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g_width = g_width
        self.num_blocks = num_blocks
        self.long_connection = long_connection
        self.stochastic_depth_drop_factor = stochastic_depth_drop_factor
        self.global_layer = None

        self.enc = nn.Sequential(
            NormalizeStd(first_channel_only=in_channels > 1),
            ResBlock(
                in_channels=self.in_channels,
                out_channels=self.g_width,
                activation_class=activation_class,
                normalization_class=normalization_class,
            ),
        )

        self.main_layers = []

        if downsample_x:
            self.main_layers.append(pooling_class(2, stride=2))
        for i in range(self.num_blocks):
            if i == self.num_blocks - 2 and downsample_x:
                self.main_layers.append(Upsample(scale_factor=2))
            self.main_layers.append(
                ResBlock(
                    in_channels=self.g_width,
                    out_channels=self.g_width,
                    activation_class=activation_class,
                    normalization_class=normalization_class,
                    pooling_class=pooling_class,
                    squeeze=squeeze,
                )
            )

        if global_layer:
            self.global_layer = GlobalLayer(
                self.g_width,
                box_size=32,
                patch_size=patch_size,
                activation_class=activation_class,
            )
        if activation_block:
            self.main_layers.append(
                AttentionBlock(
                    in_channels=self.g_width,
                    out_channels=self.g_width,
                    patch_size=patch_size,
                    activation_class=activation_class,
                    normalization_class=normalization_class,
                )
            )

        self.main_layers = nn.ModuleList(self.main_layers)
        self.skip_layer = nn.Identity()

        self.last_layers = [
            RegularBlock(
                in_channels=self.g_width * (2 if global_layer else 1)
                + (in_channels if self.long_connection else 0),
                out_channels=self.out_channels,
                feature_size=self.g_width,
                kernel_size=(3, 3),
                padding=(1, 1),
                activation_class=activation_class,
                normalization_class=normalization_class,
                no_last_act=True,
            )
        ]
        if max_neighbour:
            self.last_layers.append(MaxNeighbour(2, 2))
        if end_activation_class is not None:
            self.last_layers.append(end_activation_class())
        self.last_layers = nn.Sequential(*self.last_layers)

    def forward(self, x):
        identity = x
        x = self.enc(x)
        for module in self.main_layers:
            if not (
                self.training and np.random.rand() < self.stochastic_depth_drop_factor
            ):
                x = module(x)
        if self.global_layer is not None:
            x = torch.cat((self.global_layer(x), x), dim=1)
        if self.long_connection:
            return self.last_layers(torch.cat((identity, x), dim=1))
        else:
            return self.last_layers(x)
