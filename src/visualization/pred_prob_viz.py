"""
@author: Saurav Bose
"""

import os,time
from os import listdir
from os.path import isfile, join
import random
from collections import Counter

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches


from PIL import Image
import SimpleITK as sitk

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

metadata_path = '../../data/metadata/validation_112_CORRECTED.csv'
img_path = '../../data/processed_sample/validation/'
result_save_path = '../../reports/figures/'

#Define the best model
mode = 'OTSFT'
train_meta = 'training'
val_meta = 'validation_112_CORRECTED'

#Name of and path to best model result file
file_name = f'Resnet50_{mode}_numGPU_1_lr_0.0001_wd_0.0_bs_16_numEpoch_50_train_{train_meta}_val_{val_meta}_predictions.pth'
file_path = '../results/no_jitter/'


predictions = torch.load(file_path + file_name,map_location=torch.device('cpu'))
pred_prob = predictions['prediction_probabilities'][:,0]

#Get metadata
validation_df = pd.read_csv(metadata_path)
validation_df = validation_df.loc[validation_df.label!=1].reset_index(drop=True)
validation_df["label"].replace({2: 1}, inplace=True)
validation_df['pred_prob'] = pred_prob


#Image prediction_probabilities
max_prob_pred = validation_df.groupby('image').agg({'label':'max', 'pred_prob':'max'}).reset_index()

#Youden stat = sensitivity + specificity -1 = TPR - FPR
def youden_threshold(labels, pred_prob):
    fpr, tpr, thresholds = roc_curve(labels, pred_prob)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


# optimal_threshold = youden_threshold(validation_df["label"].values, pred_prob)
optimal_threshold = youden_threshold(max_prob_pred.label, max_prob_pred.pred_prob) #Based on image probs

def plot_graphic():

    LABEL_FONT_SIZE = 16
    TITLE_FONT_SIZE = 24
    LEGEND_FONT_SIZE = 16
    TICK_FONT_SIZE = 14
    MULTI_FIG_SIZE = (24, 24)
    SINGLE_FIG_SIZE = (10,8)
    MARKER_SIZE = 10

    #Define color map
    cvals  = [0., optimal_threshold, 1.0]
    colors = ["green", "yellow", "orange"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


    img_name_arr = [f for f in listdir(img_path)]

    for img_name in img_name_arr:
        print(f'Working on {img_name}')

        img_res = validation_df.loc[validation_df.image == img_name][['image','patch_xmin','patch_ymin', 'label', 'pred_prob']]
        max_prob = max(img_res['pred_prob'])

        img_sitk = sitk.ReadImage(os.path.join(img_path, img_name))
        img_arr = sitk.GetArrayViewFromImage(img_sitk)

        x_min_arr = [0,112,224,336,448]
        y_min_arr = [0,112,224,336,448]

        patch_list = []
        flag_arr = []
        counter = 0

        for x_min in x_min_arr:
            for y_min in y_min_arr:
                img_patch = img_sitk[x_min:x_min + 224, y_min:y_min + 224, 0]
                patch_arr = sitk.GetArrayViewFromImage(img_patch).astype('float32')
                patch_list.append(patch_arr)

                if counter >= len(img_res):
                    flag_arr.append((0,'n/a'))

                elif (x_min == img_res.iloc[counter]['patch_xmin']) & (y_min == img_res.iloc[counter]['patch_ymin']):
                    flag_arr.append((1,img_res.iloc[counter]['pred_prob']))
                    counter+=1
                else:
                    flag_arr.append((0,'n/a'))

        fig = plt.figure(figsize=MULTI_FIG_SIZE)

        #Plot whole image
        ax1 = fig.add_subplot(121)
        ax1.imshow(img_arr[0], cmap = 'gray')

        #Plot image grid
        grid = ImageGrid(fig, 122,  # similar to subplot(111)
                         nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                         axes_pad=0.1,
                         direction  = 'column'# pad between axes in inch.
                         )

        for idx, (ax, im) in enumerate(zip(grid, patch_list)):
            if flag_arr[idx][0]==1:
                color = cmap(flag_arr[idx][1])
                if (flag_arr[idx][1] == max_prob) & (max_prob >= optimal_threshold):
                    color = 'red'
                rect = patches.Rectangle((0, 0),224, 224, linewidth=10, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            # Iterating over the grid returns the Axes.
            ax.imshow(im, cmap = 'gray')

        fig.suptitle(img_name, fontsize=TITLE_FONT_SIZE, y = 0.75, fontweight="bold");


        fig.savefig(result_save_path+ img_name.split('.')[0] + '_pred_viz.pdf', dpi=300, bbox_inches='tight', pad_inches=.15)
        plt.close()
plot_graphic()
