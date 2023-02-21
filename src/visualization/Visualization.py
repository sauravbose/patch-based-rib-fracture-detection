# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 21:27:03 2021

@author: danie
"""

#%% Import Required Packages 
import os 
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import requests
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from xml.dom import minidom
import shutil
import cv2
import seaborn as sns
#%% PATHS 
dir_path = os.path.dirname(os.path.realpath(__file__))

#%% Functions for defining the bounding boxes
def main(dir_path):
    os.chdir(dir_path)
    os.chdir('../')
    os.chdir('../')
    base = os.getcwd()
    data = pd.read_csv(os.path.join(base, 'data', 'interum', 'SegmentedData',
                                    'RibFracture_BB.csv'))
    print(data)
    
    class_data = pd.read_csv(os.path.join(base, 'data', 'interum', 
                                          'SegmentedData',
                                          'AP_Chest_BoundingBoxes.csv'))
    
    class_data_sub = class_data[['image', 'x_orig','y_orig',
                                'xmin', 'xmax', 
                                'ymin','ymax',
                                'x_crop','y_crop']]
    
    combine_cropped = data.merge(class_data_sub, on='image', how='left')
    print(combine_cropped.columns)
    
    val_counts = data.image.value_counts().values
    ax = sns.histplot(x=val_counts, binwidth=0.6, kde=True)
    ax.set(xlabel='#  of Rib Fractures per Radiograph', ylabel='Count')
    
    print('Mean fracture per image: ', np.mean(val_counts))
    print('Std: ', np.std(val_counts))
    print('Min-Max: ', np.min(val_counts), np.max(val_counts))
    
    combine_cropped['xdim'] = combine_cropped['x_max'] - combine_cropped['x_min']
    combine_cropped['ydim'] = combine_cropped['y_max'] - combine_cropped['y_min']
    ax2 = sns.jointplot(data=combine_cropped, x="xdim", y="ydim")
    #ax2.set(title  = 'Rib Fracture Bounding Box dimensions')
    ax2.fig.suptitle('Rib Fracture Bounding Box dimensions')
    ax2.fig.tight_layout()
    ax2.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
    
    print(combine_cropped.xdim.describe())
    print(combine_cropped.ydim.describe())
    print(combine_cropped.x_crop.describe())
    print(combine_cropped.y_crop.describe())
    
    #single_image = data[data.image == '2817263_UdZvs.nii']
    #print(single_image)
    # Show a single image 
    single_image = combine_cropped[combine_cropped.image == '2817263_UdZvs.nii']
    print(single_image[['x_min', 'x_max', 'y_min']])#[.columns]

if __name__ == "__main__":
    main(dir_path)  
#%%
