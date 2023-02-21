"""
Adjust Minimum Fracture Area as Positive

Created on Wed Aug 25 11:15:20 2021
@author: pattondm

Description: Adjust the minimum acceptable fracture area present to consider 
the patch as positive for fracture (Change from 0.60 --> 0.90) 

Current training Distribution:
# 3x3 grid (Step size= 224, p = 0.60): 0 --> 4368; 2 --> 328 
# 3x3 grid (Step size= 224, p = 0.90): 0 --> 4368; 2 --> 283 

"""
import os
import pandas as pd
import numpy as np
#%% PATHS
path = "Z:\\RibFractureStudy\\data\\processed\\train_val"
noribfrac_path = "Z:\\RibFractureStudy\\data\\processed\\training\\NoRibFractureData"
save_path = "Z:\\RibFractureStudy\\data\\processed"

#%% MAIN
training_df = 'training.csv'
training = pd.read_csv(os.path.join(save_path, training_df))

print("Original Distribution (area = 0.60")
print(training.label.value_counts())


training.loc[(training['max_fracture_area'] < 0.90) & (training['label'] == 2.0),
             'label'] = 1

print("New Distribution (area = 0.90")
print(training.label.value_counts())

print('Number of poasitive removed as positive for fracture: ', 328 -\
      training.label.value_counts()[2])

training.to_csv(os.path.join(save_path, 'training_0.90.csv'))










