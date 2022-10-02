import torch
import torch.nn as nn

from model_angelo.models.common_modules import FcResBlock
from model_angelo.utils.affine_utils import affine_from_3_points, affine_mul_vecs


class BackboneFrameNet(nn.Module):
    def __init__(self, in_features: int, activation_class=nn.ReLU) -> None:
        super().__init__()
        self.hfz = in_features
        self.embed_fc = nn.Sequential(
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
            FcResBlock(self.hfz, self.hfz, activation_class=activation_class),
        )
        self.backbone_fc = nn.Linear(self.hfz, 9)

        torch.nn.init.normal_(self.backbone_fc.weight, std=0.02)
        self.backbone_fc.bias.data = torch.Tensor([0, 1, 0, 0, 0, 0, 1, 0, 0])

    def forward(self, x, affine):
        ncac = self.backbone_fc(self.embed_fc(x)).reshape(-1, 3, 3)
        ncac = affine_mul_vecs(affine, ncac)
        new_affine = affine_from_3_points(
            ncac[..., 2, :], ncac[..., 1, :], ncac[..., 0, :]
        )
        # Mirror x and z axis
        new_affine[..., :, 0] = new_affine[..., :, 0] * (-1)
        new_affine[..., :, 2] = new_affine[..., :, 2] * (-1)
        return ncac, new_affine
