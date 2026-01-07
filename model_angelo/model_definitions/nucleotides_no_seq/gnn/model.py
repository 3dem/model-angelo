import torch.nn as nn
from model_angelo.gnn.multi_layer_ipa_no_seq import MultiLayerSeparableIPANoSeq


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel_size = 1.0
        self.ipa = MultiLayerSeparableIPANoSeq(
                1280, 
                256, 
                num_layers=8,
                q_length=7,
                p_context=23,
        )
    def forward(self, *args, **kwargs):
        return self.ipa(*args, **kwargs)
    @property
    def training_record(self):
        return self.ipa.training_record
