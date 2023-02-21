"""
# Rib Fracture AP Chest Crop and Resize

Last Edited: 9/9/2021
Author: Daniella Patton

Description:

Crops the full size radiograph using the AP chest bounding box and downsizes
the any specificed size. The rib fracture bounding box data is also
updated to accuratly reflect the rib fracture location on the downsized image.
"""
#%% PACKAGES
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
pd.options.mode.chained_assignment = None

#%% PATHS
save_img_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/test_data/nii_crop_resize/RibFractureData'
metadata_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/metadata/test_set'
img_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/test_data/RibFractureData/nii'


downsize = 672 # Downsizing the input image in pixel size
#%% FUNCTIONS
def resize_nii(nii_im_cropped_arr, df, downsize):
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
    #Creating a black square with NUMPY
    f = np.zeros((s,s))
    #Getting the centering position
    ax, ay = (s - resize_im.shape[1])//2, (s - resize_im.shape[0])//2

    df['xmin_frac_resized'] = (df['xmin_frac']/ratio).astype(int) + ax
    df['xmax_frac_resized'] = (df['xmax_frac']/ratio).astype(int) + ax
    df['ymin_frac_resized'] = (df['ymin_frac']/ratio).astype(int) + ay
    df['ymax_frac_resized'] = (df['ymax_frac']/ratio).astype(int) + ay

    #Pasting the 'image' in a centering position
    f[ay: ay + resize_im.shape[0], ax: resize_im.shape[1] + ax] = resize_im
    return ratio, f, df

def plot_images(img, df, image_type):
    '''
    image_type: 'cropped' , 'resized'
    '''
    plt.figure()
    if image_type == 'cropped':
        plt.imshow(img[0,], cmap ='gray')
        for index, row in df.iterrows():
            plt.plot((int(row['xmin_frac']), int(row['xmin_frac']), int(row['xmax_frac']),
                  int(row['xmax_frac']), int(row['xmin_frac'])),
                 (int(row['ymin_frac']), int(row['ymax_frac']), int(row['ymax_frac']),
                  int(row['ymin_frac']), int(row['ymin_frac'])),
                 color='red', markersize=100)
    if image_type == 'resized':
        plt.imshow(img, cmap = 'gray')
        for index, row in df.iterrows():
            plt.plot((int(row['xmin_frac_resized']), int(row['xmin_frac_resized']), int(row['xmax_frac_resized']),
                  int(row['xmax_frac_resized']), int(row['xmin_frac_resized'])),
                 (int(row['ymin_frac_resized']), int(row['ymax_frac_resized']), int(row['ymax_frac_resized']),
                  int(row['ymin_frac_resized']), int(row['ymin_frac_resized'])),
                 color='red', markersize=100)
    plt.show()

#%% MAIN
def main(metadata_path, save_img_path, downsize):
    data = pd.read_csv(os.path.join(metadata_path, 'RibFracture_BB.csv'))

    class_data = pd.read_csv(os.path.join(metadata_path, 'APChestBB_Fracture.csv'))
    class_data_sub = class_data[['image', 'x_orig','y_orig',
                                'xmin', 'xmax',
                                'ymin','ymax',
                                'x_crop','y_crop']]

    class_data_sub['image'] = class_data_sub['image'].apply(lambda x: x.split('.')[0]).values

    combine_cropped = data.merge(class_data_sub, on='image', how='left')

    unique_image_list = combine_cropped['image'].unique()
    i = 0
    for unique_image in unique_image_list:
        i = i + 1

        # Show a single image for example
        single_im = combine_cropped[combine_cropped['image'] == unique_image]
        single_im = single_im.reset_index(drop=True)
        single_im = single_im.loc[:, ~single_im.columns.str.contains('^Unnamed')]

        # Crop the Image Base don .xml file
        nii_im = sitk.ReadImage(os.path.join(img_path, single_im.image[0]+'.nii'))
        spacing_orig = nii_im.GetSpacing()
        nii_arr = sitk.GetArrayViewFromImage(nii_im) #Note that Nifty img arr is channels x y_dim x x_dim

        # Plot the cropped image
        xmin_chest, ymin_chest = int(single_im.xmin[0]), int(single_im.ymin[0])
        xmax_chest, ymax_chest = int(single_im.xmax[0]), int(single_im.ymax[0])
        nii_arr_c = nii_arr[:,ymin_chest:ymax_chest,xmin_chest:xmax_chest]

        # Finding the the nex x-min and x-max relative to the chest crop
        single_im['xmin_frac'] = single_im['x_min'] - single_im['xmin']
        single_im['xmax_frac'] = single_im['x_max'] - single_im['xmin']
        single_im['ymin_frac'] = single_im['y_min'] - single_im['ymin']
        single_im['ymax_frac'] = single_im['y_max'] - single_im['ymin']

        # Plot img
        # plot_images(nii_arr_c, single_im, 'cropped')

        # Resize Image
        ratio, resize_672, resized_df = resize_nii(nii_arr_c, single_im, downsize)
        image_672 = sitk.GetImageFromArray(np.expand_dims(resize_672, axis = 0))
        image_672.SetSpacing((spacing_orig[0]*ratio, spacing_orig[1]*ratio, 1))
        path = os.path.join(save_img_path, single_im['image'][0]+'.nii')
        sitk.WriteImage(image_672, path)

        try:
            frames = [final_672_df, resized_df]
            final_672_df = pd.concat(frames)
        except:
            final_672_df = resized_df


        df_to_save = final_672_df[['image', 'n', 'acuity', 'loc', 'cert',
                                   'xmin_frac_resized', 'xmax_frac_resized',
                                   'ymin_frac_resized', 'ymax_frac_resized']]
        df_to_save.to_csv(os.path.join(metadata_path, 'RibFracture_Cropped_Resized.csv'))

if __name__ == "__main__":
    main(metadata_path, save_img_path, downsize)
