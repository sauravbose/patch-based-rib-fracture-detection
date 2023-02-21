# -*- coding: utf-8 -*-
"""
Rib Fracture Data Organization 
Date: 2/26/2021
@author: Daniella Patton

This .py file reorganizes the rib fracture dataset dowloaded using pMRI post
running the dcm_pMRI_to_patientfolder_organization.py file.

Original Organization:
---Data
    --- patient folder
            --- dcm file
            --- dcm file
            --- dcm file
            
New Organization:
---Data
    --- accession_num.nii
    --- accession_num.nii
    --- accession_num.nii
"""
#%% REQUIRED PACKAGES
# Python Version: 3.8.5
import os
import glob
import shutil
import pandas as pd # 1.1.3
import SimpleITK as sitk # 2.0.2
import string, random
from os.path import exists

#%% PATH 
# Specifiy the base dir where originaldowloaded pMRI folders are stored
#base_dir = "Z:\\RibFractureStudy\\data\\Study2_Processed_nii_2\\NoRibFractureData"
base_dir = "Z:\\RibFractureStudy\\data\\examples\\Study2_Processed_nii_2\\ex2"
#%% MAIN
def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


#%%
def main(base_dir):
    unconverted_files = []
    
    # Find all patient folders in the base_dir
    patient_folders = [f.path for f in os.scandir(base_dir) if f.is_dir() ]
    for patient_folder in patient_folders:
        # find the accession number from the folder name 
        accession_num = os.path.basename(patient_folder)
        # change the directory
        os.chdir(patient_folder)
        i = 0
        # find all dcm files in each dcm subfolder 
        for dcm_image in [f for f in glob.glob("*.dcm")]:
            i = i + 1
            # Define new name to to convert dcm to nifti
            nifti_name = accession_num + '_' + id_generator(5) + '.nii' 
            
            # Check if the file exists in folder
            if exists(os.path.join(base_dir, nifti_name)):
                nifti_name = accession_num + '_' + id_generator(5) + '.nii'     
    
            try:
                # Convert the .dcm to .nii
                reader = sitk.ImageFileReader()
                reader.SetFileName(dcm_image)
                image = reader.Execute()
                writer = sitk.ImageFileWriter()
                writer.SetImageIO("NiftiImageIO")
                writer.SetFileName(nifti_name)
                writer.Execute(image)
    
                # move the .nii file to the patient folder
                src = os.path.join(patient_folder, nifti_name)
                dst = os.path.join(base_dir, nifti_name)
                shutil.move(src, dst)
                os.remove(dcm_image)
            except:
                unconverted_files.append([patient_folder, dcm_image])
    
    unconverted_df = pd.DataFrame(unconverted_files, columns = ['dir', 'dcm'])
    print(unconverted_df)
    # Remove empty dcm subfolders
    # NOTE: os will only remove a folder if is completely empty
    # Remove dcm_subfolder 
    
    os.chdir(base_dir)
    # find all patient folders in the base_dir
    patient_folders = [f.path for f in os.scandir(base_dir) if f.is_dir() ]
    for patient_folder in patient_folders:
        os.rmdir(patient_folder)

if __name__ == '__main__':
    main(base_dir)