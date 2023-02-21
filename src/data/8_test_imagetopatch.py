"""
# Test set --> Patches

Last Edited: 10/26/2021
Author: Saurav Bose, Daniella Patton

Description:
Splitting the data into training and validation sets (80% training, 20% validation)
and creating a simple 3 x 3 rid for patches. The data is split by consider the
unique accession number.
"""
#%% PACKAGES
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import matplotlib.pyplot as plt
# import seaborn as sns
import glob

from func.helper import plot_images, calculate_area, final_csv_creater, norib_final_csv_creater

#%% PATHS
metadata_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/metadata/test_set'
img_path = '/Users/boses1/Personal/DataScience/Projects/RibFracture/data/test_data/nii_crop_resize'


#%% MAIN FUNCTION
def main():

    test_frac_data = pd.read_csv(os.path.join(metadata_path, 'RibFracture_Cropped_Resized.csv'))

    # Creating Patches from the Original Dataset
    test_frac_patches = final_csv_creater(test_frac_data, 112)
    test_frac_patches['Rad Type'] = 'Y'

    # Creating No Rib Fracture
    os.chdir(os.path.join(img_path, 'NoRibFractureData'))

    norib_list = [file for file in glob.glob("*.nii")]
    #print('Number of no rib images: ', len(norib_list))
    norib_df = pd.DataFrame(norib_list, columns = ['image'])
    norib_df['acc'] = norib_df.image.str[:-10]

    test_no_frac_patches = norib_final_csv_creater(norib_df, 112)
    test_no_frac_patches['Rad Type'] = 'N'

    # Concatentaing the training rib fracture and no rib fracture df to make
    # our final test set
    test_patches = pd.concat([test_frac_patches, test_no_frac_patches])

    # Save the training and validation dataset as csv
    test_patches.to_csv(os.path.join(metadata_path, 'test.csv'))

if __name__ == "__main__":
    main()
