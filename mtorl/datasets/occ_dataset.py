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
import math
import os
from collections import OrderedDict

import h5py
import torch
from mtorl.datasets import occ_transforms
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from mtorl.utils.log import logger


def _read_labels_from_png(png_files):
    edge_label_file, ori_label_file = png_files
    edge_label = read_image(edge_label_file, ImageReadMode.GRAY).float().div(255)
    ori_label = read_image(ori_label_file, ImageReadMode.GRAY).float().div(255)
    ori_label = ori_label * (2 * math.pi) - math.pi
    labels = torch.cat([edge_label, ori_label], dim=0)
    return labels


def _read_labels_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        labels = f['label'][0]
    return torch.from_numpy(labels).float()


class OccDataset(torch.utils.data.Dataset):

    def __init__(self,
                 images: list = None,
                 labels_files: list = None,
                 labels_file_type: str = 'png',
                 random_corp_size=0,
                 random_rotation_degrees=None):
        self.images = images
        self.labels_files = labels_files

        if labels_file_type == 'png':
            self.read_labels = _read_labels_from_png
        else:  # labels_file_type == 'h5'
            self.read_labels = _read_labels_from_h5

        transform_list = []
        if random_rotation_degrees is not None:
            transform_list.append(
                occ_transforms.RandomRotation(random_rotation_degrees)
            )
        if random_corp_size != 0:
            transform_list.append(
                occ_transforms.RandomCrop((random_corp_size, random_corp_size))
            )

        transform_list.append(occ_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transform_list)
        logger.info(f"Transform: {self.transform}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(self.images[idx]).float().div(255)
        labels = self.read_labels(self.labels_files[idx])
        image_name = os.path.splitext(os.path.basename(self.images[idx]))[0]

        sample = {'image': image, 'labels': labels}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['image_name'] = image_name
        return sample
