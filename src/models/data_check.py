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

metadatapath = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/metadata/validation.csv'
imagepath = '/mnt/isilon/prarie/RibFractureStudy/data/processed/train_val/data'

metadata = pd.read_csv(metadatapath)
metadata = metadata.drop('Unnamed: 0', axis = 'columns')
metadata = metadata[metadata.label != 1]

print(metadata.shape)

counter = 0

for idx in range(len(metadata)):
    img_meta = metadata.iloc[idx]

    try:
        img_sitk = sitk.ReadImage(os.path.join(imagepath, img_meta['image']))
        counter+=1

    except:
        print(img_meta['image'],' file not found')
        pass

print(counter)

# img_meta = metadata.iloc[0]
# print(img_meta['image'])
# try:
#     img_sitk = sitk.ReadImage(os.path.join(imagepath + 'RibFractureData/', img_name))
# except NameError:
#     img_sitk = sitk.ReadImage(os.path.join(imagepath + 'NoRibFractureData/', img_name))
# except:
#     print('file not found')
#
# try:
#     img_sitk = sitk.ReadImage(os.path.join(imagepath + 'NoRibFractureData/', img_name))
# except:
#     print('file not found')
