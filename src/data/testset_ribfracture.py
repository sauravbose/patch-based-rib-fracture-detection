# -*- coding: utf-8 -*-
"""
# Creating the Rib Fracture Test Set (n= 85)

Last Edited: 8/9/2021
Author: Daniella Patton

Description:
Creates the rib frcature test set (n=85) after corss checking for no overlap
with training data according to what has been segmented, the accessio number, 
and the MRN number.    
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

file_path, tail = os.path.split(__file__)
print(file_path)

#%% PATHS
seg_dir = "Z:\\RibFractureStudy\\data\\interum\\SegmentedData\\nii"
allrib_dir = 'Z:\\RibFractureStudy\\data\\interum\\ClassificationData'
patientdata_dir = 'Z:\\RibFractureStudy\\data\\Study2'
testset_dir = 'Z:\\RibFractureStudy\\data\\processed\\test\\RibFractureData'
bbdata_dir = 'Z:\\RibFractureStudy\\data\\interum\\ChestBBData'

#%% MAIN
def main(seg_dir, allrib_dir, patientdata_dir, testset_dir, bbdata_dir):
    # Remove any overlap in segmented images nii images and all images with cropped bounding boxes
    os.chdir(seg_dir)
    seg_nii = [f for f in glob.glob("*.nii")] # 352 images in our list == 340 labeled    
    ap_chest_bb = pd.read_csv(os.path.join(allrib_dir, 'AP_Chest_BoundingBoxes.csv'))
    
    # filter nii files ONLY with rib fractures
    ribfracture_only = ap_chest_bb[ap_chest_bb.label == 1] # n = 653
    
    # Remove any images that were in the segmented data list
    ribfracture_noseg = ribfracture_only[~ribfracture_only['image'].isin(seg_nii)] # n = 301
    
    # Remove any overlap by segmented accession number
    ribfracture_noseg['acc'] = ribfracture_noseg.image.str[:-10] 
    seg_acc = [e[:-10] for e in seg_nii]
    ribfracture_noseg_acc_r = ribfracture_noseg[~ribfracture_noseg['acc'].isin(seg_acc)] # n = 299
    
    # Double check and remove any overlap with MRN numbers 
    ribfracture_1 = pd.read_csv(os.path.join(patientdata_dir, 
                                             'rib_fracture.csv'))
    mrn_acc_1 = ribfracture_1[['acc', 'mrn']]
    ribfracture_2 = pd.read_csv(os.path.join(patientdata_dir, 
                                            'additional_ribfracturecases.csv'))
    mrn_acc_2 = ribfracture_2[['acc', 'mrn']]
    acc_mrn = pd.concat([mrn_acc_1, mrn_acc_2])
    del ribfracture_1, mrn_acc_1, ribfracture_2, mrn_acc_2
    
    # Double check and remove overlap with MRN numbers
    mrn_list = []
    for i in seg_acc:
        row = acc_mrn[acc_mrn['acc'] == i]
        mrn_list.append(row.mrn.values[0])
    
    ribfracture_noseg_acc_r = ribfracture_noseg_acc_r.merge(acc_mrn, on = 'acc', how = 'left')
    ribfracture_noseg_acc_r_mrn_r = ribfracture_noseg_acc_r[~ribfracture_noseg_acc_r['mrn'].isin(mrn_list)] # n = 299
    
    # Randomly select 85 for our rib fracture test set data
    rib_fracture_test_set = ribfracture_noseg_acc_r_mrn_r.sample(n = 85, random_state = random_state)
    rib_fracture_test_set = rib_fracture_test_set[['label', 'image', 'name', 'xml']]
    
    # Moving selected files to the test set directory
    os.chdir(os.path.join(testset_dir, 'nii'))
    onlyfiles = [f for f in os.listdir(os.path.join(testset_dir, 'nii')) if os.path.isfile(os.path.join(testset_dir, 'nii', f))]
    
    if len(onlyfiles) == 0:
        rib_fracture_test_set.to_csv(os.path.join(testset_dir, 'TestSet_RibFracture.csv'))
        for index, row in rib_fracture_test_set.iterrows():    
            src_im = os.path.join(allrib_dir, 'RibFracture', row['image'])
            dest_im = os.path.join(testset_dir, 'nii', row['image'])
            shutil.move(src_im, dest_im)  
            
            src_xml = os.path.join(bbdata_dir, 'xml_all', row['xml'])
            dest_xml = os.path.join(testset_dir, 'xml', row['xml'])
            shutil.move(src_xml, dest_xml)  
        
            src_png = os.path.join(bbdata_dir, 'png_all', row['name'] + '.png')
            dest_png = os.path.join(testset_dir, 'png', row['name'] + '.png')
            shutil.move(src_png, dest_png)  
            
            # Creating the Patient Report CSV for the raidologists to confirm fracture
            ribfracture_1 = pd.read_csv(os.path.join(patientdata_dir, 
                                                     'rib_fracture.csv'))
            mrn_acc_1 = ribfracture_1[['acc', 'mrn', 'text']]
            ribfracture_2 = pd.read_csv(os.path.join(patientdata_dir, 
                                                    'additional_ribfracturecases.csv'))
            mrn_acc_2 = ribfracture_2[['acc', 'mrn', 'text']]
            acc_mrn = pd.concat([mrn_acc_1, mrn_acc_2])
            del ribfracture_1, mrn_acc_1, ribfracture_2, mrn_acc_2
            
            rib_fracture_test_set['acc'] = rib_fracture_test_set.image.str[:-10]
            rib_fracture_test_set_patient_data = rib_fracture_test_set.merge(acc_mrn, on = 'acc', how = 'left')
            rib_fracture_test_set_patient_data.to_csv(os.path.join(testset_dir, 'TestSet_RibFracture_PatientReport.csv'))
    
    else:
        print('files already exist in path')

if __name__ == '__main__':
    main(seg_dir, allrib_dir, patientdata_dir, testset_dir, bbdata_dir)


