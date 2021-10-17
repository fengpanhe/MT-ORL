#!/usr/bin/env python3
# coding=utf-8
'''
Author      : Panhe Feng
Email       : fengpanhe@gmail.com
Date        : 2021-08-19
Description : 

Copyright 2021 Panhe Feng
This source code is licensed under Apache License 2.0
'''
from mtorl.datasets.occ_dataset import OccDataset
from mtorl.datasets import occ_transforms
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from mtorl.datasets.prefetcher import PreFetcher

_IMG_DIR = 'Aug_JPEGImages'
_EDGE_LABEL_DIR = 'Aug_PngEdgeLabel'
_ORI_LABEL_DIR = 'Aug_PngOriLabel'


class PIODDataset(OccDataset):

    def __init__(self,
                 dataset_dir: str = 'data/PIOD',
                 list_file='train_ids.lst',
                 **kwargs) -> None:

        with open(list_file, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]

        images = [os.path.join(dataset_dir, _IMG_DIR, f'{name}.jpg') for name in names]
        labels_files = [
            [os.path.join(dataset_dir, _EDGE_LABEL_DIR, f'{name}.png'),
             os.path.join(dataset_dir, _ORI_LABEL_DIR, f'{name}.png')]
            for name in names]

        super().__init__(images=images,
                         labels_files=labels_files,
                         labels_file_type='png',
                         **kwargs)


def get_piod_dataloader(cfg):
    if not cfg.inference:
        train_list_file = os.path.join(cfg.dataset_dir, 'train_ids.lst')
        train_dataset = PIODDataset(cfg.dataset_dir, train_list_file,
                                    random_corp_size=cfg.random_corp_size,
                                    random_rotation_degrees=eval(cfg.random_rotation_degrees))

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=cfg.batch_size,
                                      shuffle=True,
                                      num_workers=cfg.num_workers,
                                      drop_last=True)
    else:
        train_dataloader = None

    test_list_file = os.path.join(cfg.dataset_dir, 'test_ids.lst')
    test_dataset = PIODDataset(cfg.dataset_dir, test_list_file,
                               random_corp_size=0)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=cfg.num_workers)

    return {'train': PreFetcher(train_dataloader), 'test': PreFetcher(test_dataloader)}
