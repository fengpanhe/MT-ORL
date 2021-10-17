#!/usr/bin/env python
# coding=utf-8
'''
Author      : Panhe Feng
Email       : fengpanhe@gmail.com
Date        : 2021-08-18
Description : 

Copyright 2021 Panhe Feng
This source code is licensed under Apache License 2.0
'''
from collections import OrderedDict
import torch
from torch import nn


class ConvBnRelu(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, bias=False, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBnRelu, self).__init__(OrderedDict([
            ('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                     padding=padding, groups=groups, bias=bias)),
            ('bn', torch.nn.BatchNorm2d(out_planes)),
            ('relu', torch.nn.ReLU(inplace=True))
        ]))


def crop(data1, data2):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    if h1 == h2 and w1 == w2:
        return data1
    if h1 < h2 or w1 < w2:
        pad_h = (h2 - h1) // 2 + 1
        pad_h = pad_h if pad_h > 0 else 0
        pad_w = (w2 - w1) // 2 + 1
        pad_w = pad_w if pad_w > 0 else 0
        data1 = torch.nn.ConstantPad2d((pad_w, pad_w, pad_h, pad_h), 0)(data1)
        _, _, h1, w1 = data1.size()
    assert (h2 <= h1 and w2 <= w1)
    offset_h = (h1 - h2) // 2
    offset_w = (w1 - w2) // 2
    data = data1[:, :, offset_h:offset_h + h2, offset_w:offset_w + w2]
    return data


class opm(torch.nn.Module):

    def __init__(self, planes, kernel_size1=1, kernel_size2=3):
        super(opm, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, (kernel_size1, kernel_size2), padding=(
                (kernel_size1 - 1) // 2, (kernel_size2 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (kernel_size1, kernel_size2), padding=(
                (kernel_size1 - 1) // 2, (kernel_size2 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes)
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, (kernel_size2, kernel_size1), padding=(
                (kernel_size2 - 1) // 2, (kernel_size1 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (kernel_size2, kernel_size1), padding=(
                (kernel_size2 - 1) // 2, (kernel_size1 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes)
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, 3, padding=1, bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_end = ConvBnRelu(planes * 2, planes, 1)

    def forward(self, a):
        x1 = self.relu(self.conv1(a) + self.conv2(a))
        x2 = self.conv3(a)
        x = self.conv_end(torch.cat([x1, x2], 1))
        return x


class osm(nn.Module):
    def __init__(self,
                 feature_channels=[2048, 1024],
                 upsample_scale_factors=[2, 1]):
        super(osm, self).__init__()
        self.side_convs = nn.ModuleList()
        out_channels = feature_channels[-1]
        for channels, scale_factor in zip(feature_channels, upsample_scale_factors):
            layers = OrderedDict()
            layers['conv1'] = ConvBnRelu(channels, channels, 3)
            layers['conv2'] = ConvBnRelu(channels, channels, 3)
            layers['conv3'] = torch.nn.Sequential(
                conv1x1(channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )
            if scale_factor > 1:
                layers['upsample'] = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.side_convs.append(nn.Sequential(layers))

        self.conv_end = nn.Sequential(
            ConvBnRelu(out_channels * len(feature_channels), out_channels, 1),
            ConvBnRelu(out_channels, out_channels, 3),
            ConvBnRelu(out_channels, out_channels, 3)
        )

    def forward(self, inputs):
        x_list = [convs(feat) for feat, convs in zip(inputs, self.side_convs)]
        x_list = [crop(x, x_list[-1]) for x in x_list]
        x = self.conv_end(torch.cat(x_list, 1))
        return x


class res131(nn.Module):

    def __init__(self, in_planes, out_planes=0, mid_planes=0):
        super(res131, self).__init__()
        if mid_planes == 0:
            mid_planes = in_planes
        if out_planes == 0:
            out_planes = in_planes
        self.conv1 = conv1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = conv3x3(mid_planes, mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = conv1x1(mid_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.conv4 = None
        self.bn4 = None
        if out_planes != in_planes:
            self.conv4 = conv1x1(in_planes, out_planes)
            self.bn4 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.conv4 is not None:
            identity = self.conv4(identity)
            identity = self.bn4(identity)

        out += identity
        out = self.relu(out)

        return out


class BasicDecoder(nn.Module):
    def __init__(self,
                 feature_channels=[512, 256, 128, 64, 32],
                 half_feature_channels=True,
                 feature_scales=[16, 8, 4, 2, 1],
                 path_name='boundary',
                 num_end_out=1,
                 use_pixel_shuffle=False,
                 lite=False):
        '''
        path_name = 'boundary' or 'ori'
        '''
        super(BasicDecoder, self).__init__()
        assert len(feature_channels) == len(feature_scales)

        self.half_feature_channels = half_feature_channels
        self.path_name = path_name

        if self.half_feature_channels:
            self.half_features_op = nn.ModuleList()
            for num_channel in feature_channels:
                self.half_features_op.append(
                    ConvBnRelu(num_channel, num_channel // 2, 1)
                )
            feature_channels = [num_channel //
                                2 for num_channel in feature_channels]

        self.decoder_fuses = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        self.out_upsamples = nn.ModuleList()

        for idx, num_channel in enumerate(feature_channels):
            in_num_channel = num_channel if idx == 0 else num_channel * 2

            # decoder
            self.decoder_fuses.append(nn.Sequential(
                ConvBnRelu(in_num_channel, in_num_channel, 3),
                ConvBnRelu(in_num_channel, in_num_channel, 3)
            ))

            if idx + 1 < len(feature_channels):
                if path_name == 'boundary':
                    self.out_upsamples.append(nn.Sequential(
                        self.make_upsample(in_num_channel, 1, feature_scales[idx],
                                           pixel_shuffle=False)
                    ))
                self.decoder_upsamples.append(nn.Sequential(
                    self.make_upsample(in_num_channel,
                                       feature_channels[idx + 1],
                                       feature_scales[idx] // feature_scales[idx+1],
                                       pixel_shuffle=use_pixel_shuffle)
                ))

        if path_name == 'boundary':
            end_channel_num = 8 if lite else 64
            self.end = nn.Sequential(
                ConvBnRelu(feature_channels[-1] * 2, end_channel_num, 3),
                nn.Conv2d(end_channel_num, num_end_out, 1, padding=0, bias=False)
            )
        elif path_name == 'ori':
            end_channel_num = 8 if lite else 32
            self.end = nn.Sequential(
                ConvBnRelu(feature_channels[-1] * 2, end_channel_num, 1),
                ConvBnRelu(end_channel_num, end_channel_num, 3),
                nn.Conv2d(end_channel_num, num_end_out, 3, padding=1)
            )
        else:
            raise AttributeError(f"path_name is restricted among in ['boundary', 'ori'], got {path_name} instead!")

    def make_upsample(self, in_planes, out_planes, scale_factor=2, pixel_shuffle=False):
        if scale_factor == 1:
            return nn.Conv2d(in_planes, out_planes, 1, padding=0, bias=False)

        if pixel_shuffle:
            out_planes = out_planes * scale_factor ** 2
        layers = OrderedDict()
        if in_planes != out_planes or pixel_shuffle:
            layers['conv'] = ConvBnRelu(in_planes, out_planes, 1)

        if pixel_shuffle:
            layers['upsample'] = nn.PixelShuffle(scale_factor)
        else:
            layers['upsample'] = nn.Upsample(
                scale_factor=scale_factor, mode='bilinear', align_corners=True)

        return nn.Sequential(layers)

    def forward(self, features):
        assert len(features) == len(self.decoder_fuses)

        if self.half_feature_channels:
            features = [half_feat_op(feat) for half_feat_op, feat in zip(
                self.half_features_op, features)]

        outs = []

        decoder_x = None
        for idx, feature in enumerate(features):
            if idx == 0:
                decoder_x = feature
            else:
                decoder_x = crop(decoder_x, feature)
                decoder_x = torch.cat([feature, decoder_x], 1)

            decoder_x = self.decoder_fuses[idx](decoder_x)

            if idx < len(self.decoder_upsamples):
                if self.path_name == 'boundary':
                    outs.append(self.out_upsamples[idx](decoder_x))

                decoder_x = self.decoder_upsamples[idx](decoder_x)

        outs.append(self.end(decoder_x))
        return outs


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
