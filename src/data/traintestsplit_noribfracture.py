# -*- coding: utf-8 -*-
"""
Splitting no rib frcature data into training and testing splits

Last Edited: 8/9/2021
Author: Daniella Patton

Description:
Randomly selects 430 radiographs to represent our entire no rib
fracture dataset. This was done after excluding any patients with repeat 
imaging and cross-checking with MRN numbers. 85 were then selected to 
represent the test set and all files were copied over to the respectiving 
test set directory our train_val directory. 
"""

#%% PACKAGES
# Python Version: 3.8.5
import os
import glob
import shutil
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
random_state = np.random.RandomState(1942)

#%% Defining the study directories
seg_dir = "Z:\\RibFractureStudy\\data\\interum\\SegmentedData\\nii"
allrib_dir = 'Z:\\RibFractureStudy\\data\\interum\\ClassificationData'
patientdata_dir = 'Z:\\RibFractureStudy\\data\\Study2'
testset_dir = 'Z:\\RibFractureStudy\\data\\processed\\test\\NoRibFractureData'
trainingval_dir = 'Z:\\RibFractureStudy\\data\\processed\\training'
bbdata_dir = 'Z:\\RibFractureStudy\\data\\interum\\ChestBBData'

#%% Defining the Main function
def main(seg_dir, allrib_dir, patientdata_dir, testset_dir, bbdata_dir):
    ap_chest_bb = pd.read_csv(os.path.join(allrib_dir, 'AP_Chest_BoundingBoxes.csv'))
    
    # filter nii files ONLY without rib fractures
    noribfracture_only = ap_chest_bb[ap_chest_bb.label == 0] # n = 660
    noribfracture_only['acc'] = noribfracture_only.image.str[:-10]
    
    #keep first duplicate value
    noribfracture_only = noribfracture_only.drop_duplicates(subset=['acc']) # n = 636
    
    # load patient data and append the mrn number (make sure no repeat patients)
    # Double check and remove any overlap with MRN numbers 
    noribfracture_patientdata = pd.read_csv(os.path.join(patientdata_dir, 'no_rib_fracture.csv'),
                                            low_memory=False)
    mrn_acc = noribfracture_patientdata[['acc', 'mrn']] # n = 918 
    noribfracture_only = noribfracture_only.merge(mrn_acc.dropna(), on = 'acc', how = 'left')
    noribfracture_only = noribfracture_only.drop_duplicates(subset=['mrn']) # n = 636
    
    # Randomly sample 430 of the rows and split into the training and validation/test splits 
    rib_fracture_random_sample = noribfracture_only.sample(n = 430)
    rib_fracture_random_sample = rib_fracture_random_sample[['label', 'image', 'name', 'xml']]
    
    # Randomly sample 85 for the test set and the remaining for the training/validation
    no_ribfracture_testset = rib_fracture_random_sample.sample(n = 85)
    no_ribfracture_training = rib_fracture_random_sample.drop(no_ribfracture_testset.index)
    print(len(no_ribfracture_testset), len(no_ribfracture_training))
    
    # Moving selected files to the test set directory
    os.chdir(os.path.join(testset_dir, 'nii'))
    onlyfiles = [f for f in os.listdir(os.path.join(testset_dir, 'nii')) if os.path.isfile(os.path.join(testset_dir, 'nii', f))]
    
    if len(onlyfiles) == 0:
        no_ribfracture_testset.to_csv(os.path.join(testset_dir, 'TestSet_NoRibFracture.csv'))
        no_ribfracture_training.to_csv(os.path.join(trainingval_dir, 'TrainingSet_NoRibFracture.csv'))
        
        for index, row in no_ribfracture_testset.iterrows():    
            src_im = os.path.join(allrib_dir, 'NoRibFracture', row['image'])
            dest_im = os.path.join(testset_dir, 'nii', row['image'])
            shutil.move(src_im, dest_im)  
            
            src_xml = os.path.join(bbdata_dir, 'xml_all', row['xml'])
            dest_xml = os.path.join(testset_dir, 'xml', row['xml'])
            shutil.move(src_xml, dest_xml)  
        
            src_png = os.path.join(bbdata_dir, 'png_all', row['name'] + '.png')
            dest_png = os.path.join(testset_dir, 'png', row['name'] + '.png')
            shutil.move(src_png, dest_png)  
            
        else:
            print('files already exist in path')
    
if __name__ == '__main__':
    main(seg_dir, allrib_dir, patientdata_dir, testset_dir, bbdata_dir)


