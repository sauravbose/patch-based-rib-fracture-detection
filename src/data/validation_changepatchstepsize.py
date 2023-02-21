# -*- coding: utf-8 -*-
"""
# Changing the validation patches steps size from 224 to 112

Last Edited: 8/9/2021
Author: Daniella Patton

Description:
Using our defined valiation set and infromation availble on the rib 
fracture bounding box csv (RibFracture_Cropped_Resized.csv') created a new 
validation set that has a smaller step size. 
"""
#%% PACKAGES
import os 
import pandas as pd
pd.options.mode.chained_assignment = None
file_path, tail = os.path.split(__file__)
#file_path = '/mnt/isilon/prarie/pattondm/RibFracture/src/data'
#print(file_path)
os.chdir(file_path)
from func.helper import final_csv_creater, norib_final_csv_creater
#%% PATHS 
path = "Z:\\RibFractureStudy\\data\\processed\\train_val"
noribfrac_path = "Z:\\RibFractureStudy\\data\\processed\\training\\NoRibFractureData"
save_path = "Z:\\RibFractureStudy\\data\\processed"

#%% MAIN
def main(path, noribfrac_path, save_path):
    # Read in two dataframes to update our validation data with new step size
    val_orig = pd.read_csv(os.path.join(save_path, 'validation.csv'))
    ribfrac_bb_data = pd.read_csv(os.path.join(path,'RibFracture_Cropped_Resized.csv'))
    
    # Split the original validation dataframe - rib and no rib fracture data
    val_orig_ribfract = val_orig[val_orig['Rad Type'] == 'Y']
    val_orig_noribfract = val_orig[val_orig['Rad Type'] == 'N']
    
    # Find the Unique images for our rib and no rib fracture data
    unique_images_ribfract = val_orig_ribfract.image.unique()
    unique_images_noribfract = val_orig_noribfract.image.unique()
    
    # Filtered rib fracture bb data data for images present in our validation data
    ribfrac_bb_data = ribfrac_bb_data[ribfrac_bb_data.image.isin(unique_images_ribfract)]
    
    # Making the final rib fracture dataset
    step_size = 112
    val_ribfrac_fin = final_csv_creater(ribfrac_bb_data, step_size)
    
    # Making the final no rib fracture dataset
    unique_images_noribfract_df = pd.DataFrame(unique_images_noribfract, columns = ['image'])
    val_noribfrac_fin = norib_final_csv_creater(unique_images_noribfract_df, step_size)
    
    # Add in two extra columns
    val_ribfrac_fin['dir'], val_noribfrac_fin['dir'] = path, noribfrac_path
    val_ribfrac_fin['Rad Type'], val_noribfrac_fin['Rad Type'] = 'Y', 'N'

    # save new validation csv patches
    validation_patches_f = pd.concat([val_ribfrac_fin, val_noribfrac_fin])
    validation_patches_f.to_csv(os.path.join(save_path, 
                                             'validation_' + str(step_size) +'_CORRECTED.csv')) 

if __name__ == "__main__":
    main(path, noribfrac_path, save_path) 

