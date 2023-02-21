"""
@author: Saurav Bose
"""

import os,time
import sys, getopt, copy
import random
import dill, pickle

from PIL import Image
import SimpleITK as sitk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class GreyToDuplicatedThreeChannel:
    def __call__(self, img_tensor):

        return img_tensor.repeat(3,1,1)


class ImageData(Dataset):
    def __init__(self, metadata_path, img_path, train_mode = "train"):

        """
        Initialize variables
        Load labels and image names
        Define transforms
        """

        metadata = pd.read_csv(metadata_path)
        metadata = metadata.drop('Unnamed: 0', axis = 'columns')
        metadata = metadata[metadata.label != 1]

        self.metadata = metadata

        self.img_path = img_path
        self.image_labels = metadata.label.values

        self.train_mode = train_mode

        self.transform = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    GreyToDuplicatedThreeChannel(),
                    transforms.RandomRotation(14),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    GreyToDuplicatedThreeChannel(),
                    transforms.Resize(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }


    def __len__(self):
        """
        Get the length of the entire dataset
        """
#         print("Length of dataset is ", self.image_labels.shape[0])
        return self.image_labels.shape[0]

    def __getitem__(self, idx):
        """
        Get the image item by image_index in the metadata file
        """
        img_meta = self.metadata.iloc[idx]
        x_min, y_min = img_meta.patch_xmin, img_meta.patch_ymin

        # try:
        #     img_sitk = sitk.ReadImage(os.path.join(self.img_path + 'RibFractureData/', img_meta['image']))
        # except:
        #     img_sitk = sitk.ReadImage(os.path.join(self.img_path + 'NoRibFractureData/', img_meta['image']))


        img_sitk = sitk.ReadImage(os.path.join(self.img_path, img_meta['image']))
        img_patch = img_sitk[x_min:x_min + 224, y_min:y_min + 224, 0]
        img_arr = sitk.GetArrayViewFromImage(img_patch).astype('float32')

        image_label = img_meta['label']
        if image_label == 2:
            image_label = 1

        image_label = np.array(float(image_label))

        transformed_img = self.transform[self.train_mode](img_arr)
        sample = {'image':transformed_img, 'label':torch.from_numpy(image_label).unsqueeze(0)}

        return sample
