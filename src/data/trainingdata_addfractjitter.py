# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:15:20 2021
@author: pattondm

Description: 
Adding positive patches for rib fracture to our training data to better even 
out the distirbution of our training data. This is completed using a random 
jitter function that move a box  +/- 100 pixel from the a patch centered
around a sengle fracture. 


Current training Distribution:
# 3x3 grid (Step size= 224, p = 0.60): 0 --> 4369; 2 --> 328 # num_patch = 7 (49.8%)
# 3x3 grid (Step size= 224, p = 0.90): 0 --> 4369; 2 --> 283 # num_patch = 7 (49.6%)
"""
import os
import pandas as pd
import random
file_path, tail = os.path.split(__file__)
os.chdir(file_path)
from func.helper import calculate_area

#%% PATHS
path = "Z:\\RibFractureStudy\\data\\processed\\train_val"
noribfrac_path = "Z:\\RibFractureStudy\\data\\processed\\training\\NoRibFractureData"
save_path = "Z:\\RibFractureStudy\\data\\processed"

#%% PARAMS TO DEFINE
training_df = 'training_0.90.csv'
data = pd.read_csv(os.path.join(path, 'RibFracture_Cropped_Resized.csv'))
training = pd.read_csv(os.path.join(save_path, training_df))
num_patches = 7
min_area_present = 0.90

print(training.label.value_counts())

# Training Data (n = 573) certain fractures
print(((573 * num_patches) + 283)/((573 * num_patches)+ 283 + 4369))

#%% FUNCTIONS 
def patch_check(x_min, y_min, xy_check = None):
    '''
    Function that checks the patchs is within the limit bounds of the image
    and that this patch has not already been randomly selected.
    '''
    # Check this patch is within the limit bounds of the image (max <= 672)
    if x_min < 0: x_min = 0
    if x_min > 448: x_min = 448
    if y_min < 0: y_min = 0
    if y_min > 448: y_min = 448
        
    # Check this patch has not already been randomly selected
    if xy_check:
       repeat_check = [x_min, y_min] in xy_check
       if repeat_check: x_min = -1
    return x_min, y_min

def add_images(center_x, center_y, row, num_patches, min_area_present):
    '''
    Adds one patch centered around the bounding box of the image and num_patches - 1
    additional patches that contain the 
    
    '''
    
    return_list, xy_check = [], []
    i = 0

    rib_fracture_bb = {'x1': row['xmin_frac_resized'], 
                       'x2': row['xmax_frac_resized'],
                       'y1': row['ymin_frac_resized'], 
                       'y2': row['ymax_frac_resized']} 

    x_min = int(center_x - 112)
    y_min = int(center_y - 112)
    x_min, y_min = patch_check(x_min, y_min)
    xy_check.append([x_min, y_min])
    
    while i < num_patches:
        if i == 0:
            patch_bb = {'x1': x_min, 'x2': x_min + 224, 
                        'y1': y_min, 'y2': y_min + 224}                    
            fracture_area = calculate_area(patch_bb, rib_fracture_bb)
            return_list.append([0, row.image, i, x_min, y_min, None, fracture_area,
                        None, 2, None, 'Y'])        
        
        if i != 0:
            fracture_area = 0
            while fracture_area < min_area_present: # CHANGE TO 0.60 for ORIGINAL EXPERIMENT

                x_min_new = x_min - random.randint(-100, 100)
                y_min_new = y_min - random.randint(-100, 100)
                
                x_min_new, y_min_new = patch_check(x_min_new, y_min_new, xy_check)                
                patch_bb = {'x1': x_min_new, 'x2': x_min_new + 224, 
                            'y1': y_min_new, 'y2': y_min_new + 224}                
                fracture_area = calculate_area(patch_bb, rib_fracture_bb)
                if x_min_new == -1:
                    fracture_area = 0
            
            xy_check.append([x_min_new, y_min_new])
            return_list.append([0, row.image, i, x_min_new, y_min_new,
                                None, fracture_area, None, 2, None, 'Y'])
        i = i + 1
    colnames = ['Unnamed: 0', 'image', 'patch', 'patch_xmin', 'path_ymin',
                'num_fractures', 'max_fracture_area', 'uncertain_fract_present',
                'label', 'dir', 'Rad Type']
    additional_patches = pd.DataFrame(return_list, columns = colnames)
    return additional_patches


#%% MAIN
def main(path, noribfrac_path, save_path, data, training,
         num_patches, training_df, min_area_present):
    # Splitting into training validation and test data
    os.chdir(path)
    # Keep only certain fracture (n = 705)
    fracture_data = data[data.cert == 3]
    
    # Training images with rib fractures
    training_rib = training[training['Rad Type'] == 'Y']
    training_images = training_rib.image.unique()    
    training_fractures = fracture_data[fracture_data['image'].isin(training_images)]
    
    # Keep only certain fractures in the trianing data (n = 573)
    # for each fracture, one pacth is pulled from the center of the fracture
    # n_patch - 1 additional fracture pulled
    new_patches = pd.DataFrame()
    for i, row in training_fractures.iterrows():
        center_x = row.xmin_frac_resized + int((row.xmax_frac_resized - row.xmin_frac_resized)/2)
        center_y = row.ymin_frac_resized + int((row.ymax_frac_resized - row.ymin_frac_resized)/2)
        additional_patches = add_images(center_x, center_y, row, 
                                        num_patches, min_area_present)
        new_patches = pd.concat([new_patches, additional_patches])
        
    training_added = pd.concat([training, new_patches])
    training_added.to_csv(os.path.join(save_path, 
                                       training_df[:-4] + '_addedpatches.csv'))

if __name__ == "__main__":
    main(path, noribfrac_path, save_path, data, training, 
         num_patches, training_df, min_area_present) 