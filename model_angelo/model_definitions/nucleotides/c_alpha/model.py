from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from model_angelo.models.bottleneck import Bottleneck
from model_angelo.models.common_modules import MultiScaleConv, Normalize


class InvertedRetinaFPN(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        in_channels=1,
        out_channels=1,
        features=16,
        usual_conv_settings={"kernel_size": 3, "stride": 1, "padding": 1, "groups": 1},
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        normalization_class=Normalize,
        affine=False,
    ):
        super().__init__()
        self.features = features
        self.in_planes = self.features
        self.expansion = block.expansion
        self.activation_class = activation_class
        self.activation_fn = activation_class()
        self.affine = affine

        groups = usual_conv_settings["groups"]
            
        self.modules_dict = {}
        self.module_shapes = {}

        self.normalize = normalization_class(first_channel_only=True)
        self._make_conv_layer(
            "conv1",
            in_channels,
            self.features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

        # Downsample layers
        self._make_conv_layer(
            "downsample1",
            self.features,
            self.features,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self._make_conv_layer(
            "downsample2",
            self.features,
            self.features * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self._make_conv_layer(
            "downsample3",
            self.features * 2,
            self.features * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self._make_conv_layer(
            "downsample4",
            self.features * 4,
            self.features * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        # Bottom-up layers
        self._make_layer(
            "layer1",
            block,
            1 * self.features,
            num_blocks[0],
            stride=1,
            groups=groups,
            conv_class=conv_class,
        )
        self._make_layer(
            "layer2", block, 1 * self.features, num_blocks[1], stride=1, groups=groups, conv_class=conv_class,
        )
        self._make_layer(
            "layer3", block, 2 * self.features, num_blocks[2], stride=1, groups=groups, conv_class=conv_class,
        )
        self._make_layer(
            "layer4", block, 4 * self.features, num_blocks[3], stride=1, groups=groups, conv_class=conv_class,
        )
        

        # Translator layers
        # Takes the output of layer{i} and makes it match the output of latlayer{i}
        self._make_conv_layer("translator_layer1", self.module_shapes["layer2"]["out"], self.module_shapes["layer1"]["in"], **usual_conv_settings)
        self._make_conv_layer("translator_layer2", self.module_shapes["layer3"]["out"], self.module_shapes["layer2"]["in"], **usual_conv_settings)
        self._make_conv_layer("translator_layer3", self.module_shapes["layer4"]["out"], self.module_shapes["layer3"]["in"], **usual_conv_settings)
        self._make_conv_layer("translator_layer4", self.module_shapes["downsample4"]["out"], self.module_shapes["layer4"]["in"], **usual_conv_settings)

        # Lateral layers
        self._make_conv_layer(
            "latlayer1", self.module_shapes["conv1"]["out"], self.module_shapes["layer1"]["in"], **usual_conv_settings
        )
        self._make_conv_layer(
            "latlayer2", self.module_shapes["downsample1"]["out"], self.module_shapes["layer2"]["in"], **usual_conv_settings
        )
        self._make_conv_layer(
            "latlayer3", self.module_shapes["downsample2"]["out"], self.module_shapes["layer3"]["in"], **usual_conv_settings
        )
        self._make_conv_layer(
            "latlayer4", self.module_shapes["downsample3"]["out"], self.module_shapes["layer4"]["in"], **usual_conv_settings
        )

        # Predictor heads
        self.modules_dict["predictor"] = MultiScaleConv(self.module_shapes["layer1"]["out"], out_channels, mid_channels=self.features)
        
        self.modules_dict = nn.ModuleDict(self.modules_dict)

    def _make_conv_layer(self, name, in_features, out_features, **conv_settings):
        self.modules_dict[name] = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv3d(in_features, out_features, **conv_settings)),
                ("bn", nn.InstanceNorm3d(out_features, affine=self.affine)),
                ("act", self.activation_class())
            ])
        )
        self.module_shapes[name] = {"in": in_features, "out": out_features}

    def _make_layer(self, name, block, planes, num_blocks, stride, dilations=None, **kwargs):
        old_in_planes = self.in_planes
        strides = [stride] + [1] * (num_blocks - 1)
        if dilations is None:
            dilations = [1] * len(strides)
        layers = []
        for stride, dilation in zip(strides, dilations):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    dilation=dilation,
                    activation_class=self.activation_class,
                    **kwargs,
                )
            )
            self.in_planes = planes * block.expansion
        self.modules_dict[name] = nn.Sequential(*layers)
        self.module_shapes[name] = {"in": old_in_planes, "out": self.in_planes}

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv3d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose trilinear interpolation which supports arbitrary output sizes.
        """
        _, _, H, W, D = y.size()
        return F.interpolate(x, size=(H, W, D), mode="trilinear") + y

    def prep_features_dict(self, x, **kwargs):
        return {"x": self.normalize(x), **kwargs}

    def forward(self, x, **kwargs):
        features_dict = self.prep_features_dict(x, **kwargs)

        features_dict = self.forward_downsample(features_dict)
        features_dict = self.forward_blocks(features_dict)
        features_dict = self.forward_predictions(features_dict)

        return features_dict["p1"]

    def forward_downsample(self, features_dict):
        features_dict["ds0"] = self.modules_dict["conv1"](features_dict["x"])
        features_dict["ds1"] = self.modules_dict["downsample1"](features_dict["ds0"])
        features_dict["ds2"] = self.modules_dict["downsample2"](features_dict["ds1"])
        features_dict["ds3"] = self.modules_dict["downsample3"](features_dict["ds2"])
        features_dict["ds4"] = self.modules_dict["downsample4"](features_dict["ds3"])
        
        return features_dict


    def forward_blocks(self, features_dict):
        features_dict["c4"] = self.modules_dict["layer4"](self._upsample_add(
            self.modules_dict["translator_layer4"](features_dict["ds4"]),
            self.modules_dict["latlayer4"](features_dict["ds3"])
        ))
        features_dict["c3"] = self.modules_dict["layer3"](self._upsample_add(
            self.modules_dict["translator_layer3"](features_dict["c4"]),
            self.modules_dict["latlayer3"](features_dict["ds2"])
        ))
        features_dict["c2"] = self.modules_dict["layer2"](self._upsample_add(
            self.modules_dict["translator_layer2"](features_dict["c3"]),
            self.modules_dict["latlayer2"](features_dict["ds1"])
        ))
        features_dict["c1"] = self.modules_dict["layer1"](self._upsample_add(
            self.modules_dict["translator_layer1"](features_dict["c2"]),
            self.modules_dict["latlayer1"](features_dict["ds0"])
        ))

        return features_dict

    def forward_predictions(self, features_dict):
        # Predictions
        features_dict["p1"] = self.modules_dict["predictor"](features_dict["c1"])

        return features_dict



class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = InvertedRetinaFPN(
            Bottleneck,
            [10, 50, 20, 2],
            in_channels=2,
            out_channels=3,
            features=64,
            usual_conv_settings={"kernel_size": 3, "stride": 1, "padding": 1, "groups": 1},
            activation_class=nn.ReLU,
            affine=True,
        )
    def forward(self, x):
        return self.net(x)
