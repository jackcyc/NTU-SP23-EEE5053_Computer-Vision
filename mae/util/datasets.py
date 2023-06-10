# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from glob import glob

import numpy as np
import PIL
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms


class ContrastAdjustment():

    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, img):
        if self.factor is None:
            return transforms.functional.autocontrast(img)
        else:
            return transforms.functional.adjust_contrast(img, self.factor)


class PupilDataset(Dataset):

    def __init__(self, root, split, transform, metadata=None):
        assert split in ('train', 'val', 'test')
        self.root = root
        self.split = split
        self.transform = transform

        if metadata is None:
            if split == 'train' or split == 'val':
                self.metadata = np.loadtxt(os.path.join(root, f'metadata_{split}.txt'), dtype=str)
            elif split == 'test':
                self.metadata = sorted(glob(os.path.join(root, '*', '*')))
                self.metadata = [os.path.join(*dir.split('/')[-2:]) for dir in self.metadata if os.path.isdir(dir)]
        else:
            self.metadata = np.loadtxt(os.path.join(root, f'{metadata}.txt'), dtype=str)

        self.img_paths = [sorted(glob(os.path.join(root, dir, '*.jpg'))) for dir in self.metadata]
        self.cnts = np.cumsum([0] + [len(img_paths) for img_paths in self.img_paths])

        if self.split != 'test':
            self.labels = [np.load(os.path.join(root, dir, 'label.npy')) for dir in self.metadata]
            self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)  # will be cast to tensor.long()

    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cnts, idx, side='right') - 1
        subidx = idx - self.cnts[chunk_idx]

        img = Image.open(self.img_paths[chunk_idx][subidx]).convert('RGB')  # ViT expects RGB
        img = self.transform(img)

        if self.split == 'test':
            return img
        else:  # train, val
            label = self.labels[idx]
            return img, label

    def __len__(self):
        return self.cnts[-1]


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = args.data_path

    if is_train:
        dataset = PupilDataset(root=root, split='train', transform=transform, metadata='metadata')
        print('train on all')
    else:
        dataset = PupilDataset(root=root, split='val', transform=transform)

    print(dataset)

    return dataset


def build_dataset_inference(root):
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = PupilDataset(root=root, split='test', transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(p=0.5, interpolation=PIL.Image.BICUBIC),
            transforms.Resize((args.input_size, args.input_size), interpolation=PIL.Image.BICUBIC),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
