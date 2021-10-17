#!/usr/bin/env python
# coding=utf-8
'''
Author      : Panhe Feng
Email       : fengpanhe@gmail.com
Date        : 2021-08-19
Description :

Copyright 2021 Panhe Feng
This source code is licensed under Apache License 2.0
'''
import math

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision import transforms as tv_transforms


class Normalize(object):
    def __init__(self, mean, std):
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = mean
        self.std = std
        self.normalize = tv_transforms.Normalize(mean, std)

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        if type(image) == list:
            for i in range(len(image)):
                image[i] = self.normalize(image[i])
        else:
            image = self.normalize(image)

        return {'image': image, 'labels': labels}


class RandomCrop(tv_transforms.RandomCrop):

    def forward(self, sample):
        image, labels = sample['image'], sample['labels']

        _, height, width = image.size()
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            labels = F.pad(labels, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            labels = F.pad(labels, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        labels = F.crop(labels, i, j, h, w)
        return {'image': image, 'labels': labels}


class RandomRotation(tv_transforms.RandomRotation):

    def forward(self, sample):
        image, labels = sample['image'], sample['labels']

        image_fill = self.fill
        if isinstance(image, Tensor):
            if isinstance(image_fill, (int, float)):
                image_fill = [float(image_fill)] * F._get_image_num_channels(image)
            else:
                image_fill = [float(f) for f in image_fill]

        labels_fill = self.fill
        if isinstance(image, Tensor):
            if isinstance(labels_fill, (int, float)):
                labels_fill = [float(labels_fill)] * F._get_image_num_channels(labels)
            else:
                labels_fill = [float(f) for f in labels_fill]

        angle = self.get_params(self.degrees)

        image = F.rotate(image, angle, self.resample, self.expand, self.center, image_fill)
        labels = F.rotate(labels, angle, self.resample, self.expand, self.center, labels_fill)

        b_label, o_label = labels[0], labels[1]
        mask = b_label == 1
        o_label[mask] += (angle * math.pi / 180.0)
        o_label[mask] %= (2 * math.pi)
        labels[0], labels[1] = b_label, o_label

        return {'image': image, 'labels': labels}
