"""
# Resizig the no rib fracture data in the training and validaiton set

Last Edited: 9/9/2021
Author: Daniella Patton

Description:
Crops and resizes the no rib fracture data in the training and validation set
"""

#%% PACKAGES
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

#%% PATHS
save_img_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/test_data/nii_crop_resize/NoRibFractureData'
metadata_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/metadata/test_set'
img_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/test_data/NoRibFractureData/nii'

downsize = 672

#%% FUNCTIONS
def resize_nii(nii_im_cropped_arr, downsize):
    '''
    Parameters
    ----------
    nii_im_cropped_arr : np.ndarray
        DESCRIPTION. Cropped nifti image that will be resized
    downsize : int
        DESCRIPTION. The output size of the nifti array

    Returns
    -------
    ratio : float
        DESCRIPTION. The factor in which the niti file was scaled up or down
        by. This will be used to adjust the image spacing
    f : np.ndarray
        DESCRIPTION.A square np.ndarray that is scaled down so that the larger
        of the x, y dimnesion is == downsize. Padding is then applied to
        either side of the smaller dimension to return a square matrix
    '''
    x, y = nii_im_cropped_arr.shape[2], nii_im_cropped_arr.shape[1]
    if x >= y:
        ratio = x/downsize
        y_new = int(y/ratio)
        dsize = (downsize, y_new)
        resize_im = cv2.resize(nii_im_cropped_arr[0,], dsize, interpolation = cv2.INTER_AREA)
    if x < y:
        ratio = y/downsize
        x_new = int(x/ratio)
        dsize = (x_new, downsize)
        resize_im = cv2.resize(nii_im_cropped_arr[0,], dsize, interpolation = cv2.INTER_AREA)


    # Change range to 0 - 1
    resize_im = (resize_im - resize_im.min())*(1/(resize_im.max() - resize_im.min()))
    # Pad so that the image is a square
    #Getting the bigger side of the image
    s = max(resize_im.shape[0:2])
    #Creating a dark square with NUMPY
    f = np.zeros((s,s))
    #Getting the centering position
    ax, ay = (s - resize_im.shape[1])//2, (s - resize_im.shape[0])//2

    #Pasting the 'image' in a centering position
    f[ay: ay + resize_im.shape[0], ax: resize_im.shape[1] + ax] = resize_im
    return ratio, f

#%% MAIN
def main(metadata_path, save_img_path, downsize):

    apchestbb_csv = pd.read_csv(os.path.join(metadata_path, 'APChestBB_No_Fracture_corrected.csv'))

    # Loop through each row, load the image, and crop using the bounding box
    # data, and resize to 672 x 672 pixels

    for i, row in apchestbb_csv.iterrows():
        # Crop the Image Base don .xml file
        nii_im = sitk.ReadImage(os.path.join(img_path, row.image))
        spacing_orig = nii_im.GetSpacing()
        nii_arr = sitk.GetArrayViewFromImage(nii_im)

        # Crop the image using the AP chest bounding box
        xmin_chest, ymin_chest = int(row.xmin), int(row.ymin)
        xmax_chest, ymax_chest = int(row.xmax), int(row.ymax)
        nii_arr_c = nii_arr[:,ymin_chest:ymax_chest,xmin_chest:xmax_chest]
        # plt.imshow(nii_arr_c[0,], cmap ='gray')

        # Resize Image
        try:
            ratio, resize_672 = resize_nii(nii_arr_c, downsize)
        except:
            print(row.image)
        # plt.imshow(resize_672, cmap ='gray')

        image_672 = sitk.GetImageFromArray(np.expand_dims(resize_672, axis = 0))
        image_672.SetSpacing((spacing_orig[0]*ratio, spacing_orig[1]*ratio, 1))

        path = os.path.join(save_img_path, row.image)
        sitk.WriteImage(image_672, path)


if __name__ == "__main__":
    main(metadata_path, save_img_path, downsize)
