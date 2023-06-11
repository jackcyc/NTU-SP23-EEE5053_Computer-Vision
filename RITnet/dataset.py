#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:47:44 2019

@author: Aayush

This file contains the dataloader and the augmentations and preprocessing done

Required Preprocessing for all images (test, train and validation set):
1) Gamma correction by a factor of 0.8
2) local Contrast limited adaptive histogram equalization algorithm with clipLimit=1.5, tileGridSize=(8,8)
3) Normalization
    
Train Image Augmentation Procedure Followed 
1) Random horizontal flip with 50% probability.
2) Starburst pattern augmentation with 20% probability. 
3) Random length lines augmentation around a random center with 20% probability. 
4) Gaussian blur with kernel size (7,7) and random sigma with 20% probability. 
5) Translation of image and labels in any direction with random factor less than 20.
"""

import copy
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
  
class RandomHorizontalFlip(object):
    def __call__(self, img,label):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT),\
                        label.transpose(Image.FLIP_LEFT_RIGHT)
        return img,label

def getRandomLine(xc, yc, theta):
    x1 = xc - 50*np.random.rand(1)*(1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*np.random.rand(1) + 50)*(1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    return map(int, (x1, y1, x2, y2))

class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value=np.random.randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img,(7,7),sigma_value))

class Translation(object):
    def __call__(self, base,mask):
        factor_h = 2*np.random.randint(1, 20)
        factor_v = 2*np.random.randint(1, 20)
        mode = np.random.randint(0, 4)
#        print (mode,factor_h,factor_v)
        if mode == 0:
            aug_base = np.pad(base, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_base = aug_base[:-factor_v, :]
            aug_mask = aug_mask[:-factor_v, :]
        if mode == 1:
            aug_base = np.pad(base, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_base = aug_base[factor_v:, :]
            aug_mask = aug_mask[factor_v:, :]
        if mode == 2:
            aug_base = np.pad(base, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_base = aug_base[:, :-factor_h]
            aug_mask = aug_mask[:, :-factor_h]
        if mode == 3:
            aug_base = np.pad(base, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_base = aug_base[:, factor_h:]
            aug_mask = aug_mask[:, factor_h:]
        return Image.fromarray(aug_base), Image.fromarray(aug_mask)     
            
class Line_augment(object):
    def __call__(self, base):
        yc, xc = (0.3 + 0.4*np.random.rand(1))*base.shape
        aug_base = copy.deepcopy(base)
        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       
        
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       

  
class PupilDataset(Dataset):
    def __init__(self, root, split='train', transform=transform, metadata=None, **args):
    
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        self.table = 255.0*(np.linspace(0, 1, 256)**0.8)


        assert split in ('train', 'val', 'test')
        self.root = root
        self.split = split
        self.transform = transform
        
        if metadata is None:
            if split == 'train' or split == 'val':
                self.metadata = np.loadtxt(os.path.join(root, f'metadata_{split}.txt'), dtype=str)
            elif split == 'test':
                self.metadata = sorted(glob(os.path.join(root, '*',  '*')))
                self.metadata = [os.path.join(*dir.split('/')[-2:]) for dir in self.metadata if os.path.isdir(dir)]
        else:
            self.metadata = np.loadtxt(os.path.join(root, f'{metadata}.txt'), dtype=str)

        self.img_paths = [sorted(glob(os.path.join(root, dir, '*.jpg')), key=lambda x: int(Path(x).stem)) for dir in self.metadata]
        self.cnts = np.cumsum([0] + [len(img_paths) for img_paths in self.img_paths])

    def __len__(self):
        return self.cnts[-1]

    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cnts, idx, side='right') - 1
        subidx = idx - self.cnts[chunk_idx]

        pilimg = Image.open(self.img_paths[chunk_idx][subidx]) # grayscale
        # resize to 240, 320
        pilimg = pilimg.resize((320, 240), resample=1) # Resampling.LANCZOS

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #Fixed gamma value for      
        pilimg = cv2.LUT(np.array(pilimg), self.table)

        if self.split != 'test':
            label = Image.open(self.img_paths[chunk_idx][subidx].replace('jpg', 'png')).convert('L')
            # let where label!=0 be 2 (pupil in RITnet_v3)
            label = label.point(lambda x: 2 if x != 0 else 0)

        if self.transform is not None:
            if self.split == 'train':
                if random.random() < 0.2: 
                    pilimg = Line_augment()(np.array(pilimg))    
                if random.random() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))   
                if random.random() < 0.4:
                    pilimg, label = Translation()(np.array(pilimg),np.array(label))
                
        img = self.clahe.apply(np.array(np.uint8(pilimg)))    
        img = Image.fromarray(img)      
            
        if self.transform is not None:
            if self.split == 'train':
                img, label = RandomHorizontalFlip()(img,label)
            elif self.split == 'test':
                img = TF.adjust_brightness(img, 1.5)
                img = TF.adjust_contrast(img, 0.5)
            img = self.transform(img)    


        if self.split != 'test':
            ## This is for boundary aware cross entropy calculation
            spatialWeights = cv2.Canny(np.array(label),0,3)/255
            spatialWeights=cv2.dilate(spatialWeights,(3,3),iterations = 1)*20
            
            ##This is the implementation for the surface loss
            # Distance map for each class
            # distMap = np.stack([one_hot2dist(np.array(label)==i) for i in range(0, 3)], 0)           
        
        img_name = '-'.join(self.img_paths[chunk_idx][subidx].split('/')[-3:])[:-4]

        if self.split == 'test':
            ##since label, spatialWeights and distMap is not needed for test images
            return img,0,img_name,0,0
            
        label = MaskToTensor()(label)
        return img, label, img_name,spatialWeights, 0 #np.float32(distMap) 
    

def build_dataloader():
    training_root = '../data/trainset'
    train_dataset = PupilDataset(training_root, 'train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

    val_dataset = PupilDataset(training_root, 'val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=8)

    testing_root = '../data/testset'
    test_dataset = PupilDataset(testing_root, 'test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, _, testloader = build_dataloader()
    for batch in train_loader:
        print(len(batch))
        import ipdb; ipdb.set_trace()