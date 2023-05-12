from collections import OrderedDict

import torch
import torch.nn as nn

from model_angelo.models.common_modules import FcResBlock


class TransitionLayer(nn.Module):
    def __init__(self, hidden_features: int, checkpoint: bool = True) -> None:
        super().__init__()

        self.hfz = hidden_features

        self.transition_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "resblock1",
                        FcResBlock(self.hfz, self.hfz, activation_class=nn.GELU),
                    ),
                    (
                        "resblock2",
                        FcResBlock(self.hfz, self.hfz, activation_class=nn.GELU),
                    ),
                    (
                        "resblock3",
                        FcResBlock(self.hfz, self.hfz, activation_class=nn.GELU),
                    ),
                ]
            )
        )

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x):
        return self.transition_fc(x)

    def forward_checkpoint(self, x):
        return torch.utils.checkpoint.checkpoint(
            self.forward_normal, x, preserve_rng_state=False,
        )
