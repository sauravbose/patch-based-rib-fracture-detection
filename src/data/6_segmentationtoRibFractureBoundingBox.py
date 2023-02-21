#!/home/boses1/miniconda3/bin/python3

"""
# Rib Fracture Bounding Boxes  (.nrrd segmentation --> csv)

Last Edited: 10/24/2021
Author: Saurav Bose, Daniella Patton

Description:

Adarsh Gosh, MD (AG) manually segmented all rib frcatures present in > 340
radiographs. Each manual segmentation (a blob) was converted into a counding
box and the 'x_min', 'x_max','y_min','y_max' is stored. Additional associated
information (certainty, location, and acuity) was also manually defined by
AG and saved as a csv file.
"""
#%% PACKAGES
import os, sys, glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
import math
#%% PATHS
nrrd_path = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/test_set/RibFractureData/final_test_set_10_20_21/segmentations_renamed'
metadata_path = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/metadata/test_set'

#%% FUNCTIONS
# Functions for defining the bounding boxes
def return_names(col):
    if col =='n2':
        acuity, loc, cert = 'acuity_1', 'loc_1', 'cert_1'
    if col == 'n3':
        acuity, loc, cert = 'acuity_2', 'loc_2', 'cert_2'
    if col == 'n4':
        acuity, loc, cert = 'acuity_3', 'loc_3', 'cert_3'
    if col == 'n5':
        acuity, loc, cert = 'acuity_4', 'loc_4', 'cert_4'
    if col == 'n6':
        acuity, loc, cert = 'acuity_5', 'loc_5', 'cert_5'
    if col == 'n7':
        acuity, loc, cert = 'acuity_6', 'loc_6', 'cert_6'
    if col == 'n8':
        acuity, loc, cert = 'acuity_7', 'loc_7', 'cert_7'
    if col == 'n9':
        acuity, loc, cert = 'acuity_8', 'loc_8', 'cert_8'
    if col == 'n10':
        acuity, loc, cert = 'acuity_9', 'loc_9', 'cert_9'
    return acuity, loc, cert

def define_bounding_box(nrrd_arr, i):
    nrrd_arr = np.where(nrrd_arr != i, 0, nrrd_arr)
    nrrd_arr = np.where(nrrd_arr > 0, 1, nrrd_arr)

    x = np.sum(nrrd_arr[0,], axis=0)
    x_dim = np.nonzero(x)
    x_min, x_max = x_dim[0][0], x_dim[0][-1]

    y = np.sum(nrrd_arr[0,], axis=1)
    y_dim = np.nonzero(y)
    y_min, y_max = y_dim[0][0], y_dim[0][-1]

    return x_min, x_max, y_min, y_max

#%% MAIN
def main():

    AP_chest_fracture = pd.read_csv(os.path.join(metadata_path, 'APChestBB_Fracture.csv'))
    segmented_data = pd.read_csv(os.path.join(metadata_path, 'RibFractureDataset_DATA_2021-10-20_1236.csv'))

    ##########################Quality Checks########################################
    # Finding the length of segmented dataset
    print('The size of the segmented dataset is: ', len(segmented_data))

    seg_names = segmented_data.record_id.unique()

    # making sure that their is no difference between the list
    # Check for difference in the list no difference)
    main_list = np.setdiff1d(seg_names, AP_chest_fracture.name)
    # yields the elements in `list_2` that are NOT in `list_1`
    print(main_list, len(main_list))

    # Find the list of nrrd files and where we have a differed
    os.chdir(nrrd_path)
    files = [f for f in glob.glob("*.nrrd")]
    files = [i.split('-')[0] for i in files]
    main_list = np.setdiff1d(seg_names,files)
    # yields the elements in `list_2` that are NOT in `list_1`
    print(main_list, len(main_list))
    ################################################################################

    # Getting the bounding box data
    no_bb, ribfracture_boundingbox = [], []
    # os.chdir(os.path.join(base, 'SegmentedData', 'nii'))
    for image in segmented_data.record_id:
        i=1
        if image == 'test':
            i = 4
        nrrd = image + '-label.nrrd'
        # Nrrd Array
        nrrd_im = sitk.ReadImage(os.path.join(nrrd_path, nrrd))
        nrrd_arr = sitk.GetArrayViewFromImage(nrrd_im)

        row = segmented_data[segmented_data.record_id == image]

        #print(image, nrrd_arr.shape, len(row))
        x = 0
        for col in row.columns:
            if col[0] == 'n' and col != 'name':
                if math.isnan(row[col]) == False:

                    try:
                        x_min, x_max, y_min, y_max = define_bounding_box(nrrd_arr, i)
                        acuity, loc, cert = return_names(col)
                        ribfracture_boundingbox.append([image,
                                                    row.iloc[0][col],
                                                    row.iloc[0][acuity],
                                                    row.iloc[0][loc],
                                                    row.iloc[0][cert],
                                                    x_min, x_max,
                                                    y_min, y_max
                                                    ])
                        x = 1
                        i = i + 1
                    except:
                        print(image, i)
        if x == 0:
            no_bb.append([image, row.comment])


    # Saving to a csv file
    ribfracture_boundingbox  = pd.DataFrame(ribfracture_boundingbox,
                                            columns= ['image', 'n', 'acuity',
                                                      'loc', 'cert', 'x_min',
                                                      'x_max','y_min','y_max'])

    ribfracture_boundingbox.to_csv(os.path.join(metadata_path,'RibFracture_BB.csv'))

    no_fract_found  = pd.DataFrame(no_bb, columns= ['image', 'report'])
    no_fract_found.to_csv(os.path.join(metadata_path,'NoRibFracFound.csv'))

if __name__ == "__main__":
    main()
