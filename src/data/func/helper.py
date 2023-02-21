# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:45:09 2021

@author: danie
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Disply Image Functions
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

def patch_img(nrrd_arr):
    # Traveling across the x [z, x, y]
    patch_1_1 = nrrd_arr[0,0:224, 0:224]
    patch_1_2 = nrrd_arr[0,0:224, 224:448]
    patch_1_3 = nrrd_arr[0,0:224, 448:672]
    #  Traveling across the y
    patch_2_1 = nrrd_arr[0,224:448, 0:224]
    patch_2_2 = nrrd_arr[0,224:448, 224:448]
    patch_2_3 = nrrd_arr[0,224:448, 448:672]
    patch_3_1 = nrrd_arr[0,448:672, 0:224]
    patch_3_2 = nrrd_arr[0,448:672, 224:448]
    patch_3_3 = nrrd_arr[0,448:672, 448:672]

    #print(patch_1_1.shape, patch_1_2.shape, patch_1_3.shape)
    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(patch_1_1, cmap = 'gray')
    axs[0,1].imshow(patch_1_2, cmap = 'gray')
    axs[0,2].imshow(patch_1_3, cmap = 'gray')
    axs[1,0].imshow(patch_2_1, cmap = 'gray')
    axs[1,1].imshow(patch_2_2, cmap = 'gray')
    axs[1,2].imshow(patch_2_3, cmap = 'gray')
    axs[2,0].imshow(patch_3_1, cmap = 'gray')
    axs[2,1].imshow(patch_3_2, cmap = 'gray')
    axs[2,2].imshow(patch_3_3, cmap = 'gray')

#%% PATCH BB CODE
def calculate_area(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position of top left corner of the patch
        the (x2, y2) position of bottom right corner of the patch
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the percent of the BB in a given patch
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    area = intersection_area / float(bb2_area)
    assert area >= 0.0
    assert area <= 1.0
    return area

def final_csv_creater(data, step_size):
    final_data = []
    patch = np.arange(0, 449, step_size).tolist()
    #patch = [0, 224, 448]
    data_unique = data.image.unique()
    i = 1
    for image in data_unique:
        image_df = data[data.image == image]
        i += 1
        j = 1
        for x_min in patch:
            for y_min in patch:
                #print('Xmin Ymin: ', x_min, y_min)
                x_max, y_max = x_min + 224, y_min + 224
                #print('Xmax Ymax: ', x_max, y_max)
                uncert_fracture_pres = 'No'
                patch_bb = {'x1': x_min, 'x2': x_max, 'y1': y_min, 'y2': y_max}
                fracture = []
                for i, row in image_df.iterrows():
                    rib_fracture_bb = {'x1': row['xmin_frac_resized'],
                                       'x2': row['xmax_frac_resized'],
                                       'y1': row['ymin_frac_resized'],
                                       'y2': row['ymax_frac_resized']}


                    fracture_area = calculate_area(patch_bb, rib_fracture_bb)
                    if fracture_area > 0 and row['cert'] == 3:
                        fracture.append(fracture_area)
                    elif fracture_area > 0 and row['cert'] != 3:
                        uncert_fracture_pres = 'Yes'
                        fracture.append(0.2)

                if len(fracture) > 0:
                    max_fracture_area = max(fracture)
                else:
                    max_fracture_area = 0

                j += 1
                if max_fracture_area == 0.0: label = 0
                if (max_fracture_area > 0 and max_fracture_area <= 0.60): label = 1
                if (max_fracture_area > 0.60): label = 2

                row_data = [row['image'], j - 1, x_min, y_min, len(fracture),
                              max_fracture_area, uncert_fracture_pres, label]
                final_data.append(row_data)

    colnames = ['image', 'patch', 'patch_xmin', 'patch_ymin', 'num_fractures',
                'max_fracture_area', 'uncertain_fract_present', 'label']
    final_df = pd.DataFrame(final_data, columns = colnames)
    return final_df

def norib_final_csv_creater(data, step_size):
    final_data = []
    patch = np.arange(0, 449, step_size).tolist()
    #patch = [0, 224, 448]
    data_unique = data.image.unique()
    i = 1
    for image in data_unique:
        i += 1
        j = 1
        for x_min in patch:
            for y_min in patch:
                uncert_fracture_pres = 'No'
                j += 1
                row_data = [image, j -1, x_min, y_min, 0,
                              0, uncert_fracture_pres, 0]
                final_data.append(row_data)

    colnames = ['image', 'patch', 'patch_xmin', 'patch_ymin', 'num_fractures',
                'max_fracture_area', 'uncertain_fract_present', 'label']
    final_df = pd.DataFrame(final_data, columns = colnames)
    return final_df


def final_csv_creater_diff_patch_size(data, step_size):
    final_data = []
    patch = [0, 224, 448]
    data_unique = data.image.unique()
    i = 1
    for image in data_unique:
        image_df = data[data.image == image]
        i += 1
        j = 1
        for x_min in patch:
            for y_min in patch:
                #print('Xmin Ymin: ', x_min, y_min)
                x_max, y_max = x_min + 224, y_min + 224
                #print('Xmax Ymax: ', x_max, y_max)
                uncert_fracture_pres = 'No'
                patch_bb = {'x1': x_min, 'x2': x_max, 'y1': y_min, 'y2': y_max}
                fracture = []
                for i, row in image_df.iterrows():
                    rib_fracture_bb = {'x1': row['xmin_frac_resized'],
                                       'x2': row['xmax_frac_resized'],
                                       'y1': row['ymin_frac_resized'],
                                       'y2': row['ymax_frac_resized']}


                    fracture_area = calculate_area(patch_bb, rib_fracture_bb)
                    if fracture_area > 0 and row['cert'] == 3:
                        fracture.append(fracture_area)
                    elif fracture_area > 0 and row['cert'] != 3:
                        uncert_fracture_pres = 'Yes'
                        fracture.append(0.2)

                if len(fracture) > 0:
                    max_fracture_area = max(fracture)
                else:
                    max_fracture_area = 0

                j += 1
                if max_fracture_area == 0.0: label = 0
                if (max_fracture_area > 0 and max_fracture_area <= 0.60): label = 1
                if (max_fracture_area > 0.60): label = 2

                row_data = [row['image'], j - 1, x_min, y_min, len(fracture),
                              max_fracture_area, uncert_fracture_pres, label]
                final_data.append(row_data)

        colnames = ['image', 'patch', 'patch_xmin', 'path_ymin', 'num_fractures',
                    'max_fracture_area', 'uncertain_fract_present', 'label']
        final_df = pd.DataFrame(final_data, columns = colnames)
    return final_df
