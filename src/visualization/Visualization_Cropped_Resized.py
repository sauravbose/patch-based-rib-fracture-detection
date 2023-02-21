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
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from catscatter import catscatter
#%% PATHS 
dir_path = os.path.dirname(os.path.realpath(__file__))
#%% Functions for defining the bounding boxes
def main(dir_path):
    os.chdir(dir_path)
    os.chdir('../')
    os.chdir('../')
    base = os.getcwd()
    data = pd.read_csv(os.path.join(base, 'data', 'interum', 'SegmentedData',
                                    'nii_cropped_resized',
                                    'RibFracture_Cropped_Resized.csv'))

    
    
    # Number of Fractures per dataset
    val_counts = data.image.value_counts().values
    plt.figure()
    ax = sns.histplot(x=val_counts, binwidth=0.6, kde=True)
    ax.set(xlabel='#  of Rib Fractures per Radiograph', ylabel='Count')
    plt.show()
    print('Mean fracture per image: ', np.mean(val_counts))
    print('Std: ', np.std(val_counts))
    print('Min-Max: ', np.min(val_counts), np.max(val_counts))
    
    # Certainty
    map_dict = {1.0: "Possible", 
                2.0: "Probable", 
                3.0: "Definite"}
    data["Certainty"] = data["cert"].map(map_dict)
    cert_counts = data.Certainty.value_counts().values
    plt.figure()
    ax2 = sns.countplot(x='Certainty', data = data)
    ax2.set(xlabel='Certainty', ylabel='Count')
    plt.show()
    print(data.Certainty.value_counts())

    # Acuity
    map_dict = {1.0: "Acute", 
                2.0: "Subacute", 
                3.0: "Chronic",
                4.0: "Unknown"}
    
    data["Acuity"] = data["acuity"].map(map_dict)
    
    plt.figure()
    ax2 = sns.countplot(x='Acuity', data = data)
    ax2.set(xlabel='Acuity', ylabel='Count')
    plt.show()
    print(data.Acuity.value_counts())

    # Acuity
    map_dict = {1.0: "Anterior", 
                2.0: "Lateral", 
                3.0: "Posterior"}
    
    data["Location"] = data["loc"].map(map_dict)
    
    plt.figure()
    ax2 = sns.countplot(x='Location', data = data)
    ax2.set(xlabel='Location', ylabel='Count')
    plt.show()
    
    print(data.Location.value_counts())
    
    sns.catplot(y="Certainty", hue="Acuity", kind="count",
            palette="pastel", edgecolor=".6", data=data)
    
    
    # Filtered by 75% sure
    data_cert = data[data["cert"]== 3.0]
    data_cert['all_three'] =  data_cert['Acuity'] + ' - '  + data_cert['Location'].astype('str')
    z1 = data_cert.all_three.value_counts().to_dict() #converts to dictionary
    data_cert['Frequency'] = data_cert['all_three'].map(z1) 
    print(data_cert.all_three.value_counts()) 
    print(len(data_cert))
    
    # Catscatter
    #plot it
    plt.figure()
    colors=['green','grey','black']
    catscatter(data_cert,'Location','Acuity','Frequency', color=colors, ratio=10)
    plt.show()

if __name__ == "__main__":
    main(dir_path)    

