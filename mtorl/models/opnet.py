import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mtorl.models.backbones.resnet import resnet
from mtorl.models.basic_components import (BasicDecoder, ConvBnRelu, crop, opm,
                                           osm)
from mtorl.utils.log import logger


class OPNet(torch.nn.Module):
    def __init__(self,
                 backbone_name='resnet50',
                 num_ori_out=2,
                 use_pixel_shuffle=True,
                 orientation_path=True,
                 lite=False):
        '''
        '''
        # C64
        super(OPNet, self).__init__()
        self.orientation_path = orientation_path

        self.backbone = resnet(backbone_name, deep_stem=True)

        num_decoder_end_c = [8, 8] if lite else [16, 64]
        if self.orientation_path:
            self.ori_convolution = torch.nn.Sequential(
                ConvBnRelu(3, num_decoder_end_c[0], 1),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
                ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[1], 1)
            )
            self.ori_decoder = BasicDecoder(
                feature_channels=[1024, 256, num_decoder_end_c[1]],
                feature_scales=[16, 4, 1],
                path_name='ori',
                num_end_out=num_ori_out,
                use_pixel_shuffle=False,
            )

        self.boundary_convolution = torch.nn.Sequential(
            ConvBnRelu(3, num_decoder_end_c[0], 1),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[1], 1)
        )
        self.boundary_decoder = BasicDecoder(
            feature_channels=[1024, 512, 256, 128, num_decoder_end_c[1]],
            feature_scales=[16, 8, 4, 2, 1],
            path_name='boundary',
            num_end_out=1,
            use_pixel_shuffle=use_pixel_shuffle
        )

        self.osm = osm(feature_channels=[2048, 1024],
                       upsample_scale_factors=[2, 1])

        self.encoder_sides = nn.ModuleList()
        for num_c, l_name in zip([1024, 512, 256, 128],
                                 ['opm', 'no', 'opm', 'no']):
            layer = opm(num_c) if l_name == 'opm' else torch.nn.Sequential()
            self.encoder_sides.append(layer)

        self.fuse = torch.nn.Sequential(
            conv1x1(5, 1)
        )
        self._initialize_weights()

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = features[::-1]

        osm_x = self.osm(features[:2])

        side_features = [osm_x] + features[2:]
        assert len(side_features) == len(self.encoder_sides)
        side_features = [layer(feat) for feat, layer in zip(side_features, self.encoder_sides)]

        boundary_outsides = self.boundary_decoder(side_features + [self.boundary_convolution(inputs)])
        boundary_outsides = [crop(boundary_outside, inputs)
                             for boundary_outside in boundary_outsides]

        boundary_fuse = self.fuse(torch.cat(boundary_outsides, 1))
        boundary_outsides.append(boundary_fuse)

        orientation_x = None
        if self.orientation_path:
            orientation_x = self.ori_decoder([side_features[0], side_features[2], self.ori_convolution(inputs)])[0]

        return boundary_outsides, orientation_x

    @staticmethod
    def getOrientation(orientation_x):
        return torch.atan2(orientation_x[:, 0], orientation_x[:, 1]).unsqueeze(1)

    @staticmethod
    def getBoundary(boundary_outsides):
        return torch.sigmoid(boundary_outsides[-1])

    def load_backbone_pretrained(self, pretrained_path='data/resnet50-25c4b509.pth'):
        keys = self.backbone.load_state_dict(torch.load(pretrained_path), strict=False)
        if len(keys.missing_keys) > 0:
            logger.info(f"=> Pretrained: missing_keys [{', '.join(keys.missing_keys)}]")
        if len(keys.unexpected_keys) > 0:
            logger.info(f"=> Pretrained: unexpected_keys [{', '.join(keys.unexpected_keys)}]")
        logger.info(f'=> Backbone pretrained: loaded from {pretrained_path}\n')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
