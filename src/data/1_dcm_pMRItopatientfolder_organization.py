# -*- coding: utf-8 -*-
"""
## Rib Fracture Data Organization 
Date: 2/15/2021
@author: Daniella Patton

This file reorganizes the rib fracture dataset that was dowloaded directly 
from pMRI to a less complicated data structure.

Original Organization:
---Data
    --- patient folder -- random numbers
        --- dcm subfolder
                --- dcm subfolder 2 --- dcm file
                --- dcm subfolder 2 --- dcm file        
        --- dcm subfolder -- accession number
                --- dcm subfolder 2 --- dcm file

New Organization:
---Data
    --- patient folder
            --- dcm file
            --- dcm file
            --- dcm file
"""
#%% REQUIRED PACKAGES
# Python Version: 3.8.5
import os
import glob
import shutil
import pandas as pd # 1.1.3
import pydicom # 2.1.2

#%% PATH
# Specifiy the base dir
# RibFractureData and NoRibFractureData folders were copied over to 
# Study2_Processed processed folder
#base_dir = "Z:\\RibFractureStudy\\data\\Study2_Processed\\RibFractureData"
#base_dir = "Z:\\RibFractureStudy\\data\\examples\\Study2_Processed\\RibFractureData"
base_dir = "Z:\\RibFractureStudy\\data\\examples\\Study2_Processed\\ex2"
os.chdir("Z:\\RibFractureStudy\\data")

#%% MAIN

def main(base_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    os.chdir(base_dir)
    print(os.getcwd())
    # Move the dcm files from subfolder to base patient folder 
    # find all patient folders in the base_dir
    patient_folders = [f.path for f in os.scandir(base_dir) if f.is_dir() ]
    # loop through each patient folder in the patient folders list
    for patient_folder in patient_folders:
        # find all dcm folders in each patient folder
        dcm_folders = [f.path for f in os.scandir(patient_folder) if f.is_dir() ]
        # loop through each dcm folder in each patient folder
        for dcm_folder in dcm_folders:
            # find all dcm subfolders in each dcm folder
            dcm_subfolders = [f.path for f in os.scandir(dcm_folder) if f.is_dir() ]
            # loop through each dcm subfolder in each dcm folder
            for dcm_subfolder in dcm_subfolders:
        
                # change the directory
                os.chdir(dcm_subfolder)

                # find all dcm files in each dcm subfolder 
                for dcm_image in [f for f in glob.glob("*.dcm")]:
                    # move the dcm file to the patient folder
                    src = os.path.join(dcm_subfolder, dcm_image)
                    dst = os.path.join(patient_folder, dcm_image)
                    shutil.move(src, dst)                    

                    
    # Remove empty dcm subfolders
    # NOTE: os will only remove a folder if is completely empty.
    # Thus this code has to be run twice.
    #  1) Remove dcm_subfolder 
    #  2) Remove dcm_folder
    
    #  1) Remove dcm_subfolder 
    os.chdir(base_dir)
    for patient_folder in patient_folders:
        dcm_folders = [f.path for f in os.scandir(patient_folder) if f.is_dir() ]
        for dcm_folder in dcm_folders:
            dcm_subfolders = [f.path for f in os.scandir(dcm_folder) if f.is_dir() ]
            for dcm_subfolder in dcm_subfolders:
                os.rmdir(dcm_subfolder)
                
    #  2) Remove dcm_folder            
    for patient_folder in patient_folders:
        dcm_folders = [f.path for f in os.scandir(patient_folder) if f.is_dir() ]
        for dcm_folder in dcm_folders:
            os.rmdir(dcm_folder)
    
    
    # Read in dcm file, pull out the accession number, and rename the folder
    for patient_folder in patient_folders:
        os.chdir(patient_folder)
        for dcm_image in [f for f in glob.glob("*.dcm")]:
            if os.path.isfile(os.path.join(patient_folder, dcm_image)):
                ds = pydicom.dcmread(dcm_image)
                #print(ds)
                #Tag (0008, 0050) Accession Number
                accession_num = ds[0x08, 0x50].value
                print(accession_num)
                os.chdir(base_dir)
                os.rename(patient_folder, os.path.join(base_dir, str(accession_num)))
    
    os.chdir(base_dir)                
                    
    os.chdir('../')
    print(os.getcwd())
    
    # read in our master csv file
    data_df = pd.read_csv('rib_fracture.csv')
    # create a list of all accession numbers
    accession_folders = os.listdir(base_dir)
    # create a new column that will list True if the dcm files were successfully dowloaded, False if not
    data_df['Downloaded'] = data_df['acc'].isin(accession_folders)
    print(data_df['Downloaded'].value_counts())
    # Only 6 were studies were not downloaded properly from pacs from our rib fracture folder
    data_df.to_csv('rib_fracture_edited.csv')


if __name__ == '__main__':
    main(base_dir)
