'''
Label AP Chest Radioraphs
Author: Adarsh Ghosh, MD
Edits: Daniella Patton

Edits made: converted the code to read in dcm, to read in nii images

Displays an image in the console and waits for user feedback to label the image
as a an AP chest or not AP chest radiograph.
'''
#%% PACKAGES
import matplotlib.pyplot as plt
import os
from glob import glob
import pandas as pd
import SimpleITK as sitk
#plt.ion() <- Enable interactive mode.
# then figures will not be shown on creation  = plt.ioff()
#%% PATH
#file path for the rib dataset
SAVEPATH = "Z:\\RibFractureStudy\\data\\examples\\Study2_Processed_nii_2"
PATH = "Z:\\RibFractureStudy\\data\\examples\\Study2_Processed_nii_2\\ex2"
SAVEPATH = PATH

EXT = "*.nii"
start_num, end_num = 0, 2
#%% MAIN
all_dcm_files = [file
                  for path, subdir, files in os.walk(PATH)
                  for file in glob(os.path.join(path, EXT))]
dcm_file = pd.DataFrame(all_dcm_files, columns=['file_name'])

# writing the file paths to a PANDAS DF --> adding a new collumn to dcm_file to record chest measurements
dcm_file['Chest'] = ''

#df = pd.DataFrame(columns=['file_name','chest']) #working DF to save the file names
 # Creating a loop to iterate through all the filenames in the pandas df
# will take a yes and no as input for presence of an AP chest xray
# save that as an additional column in the duplicate DF

dcm_file_subset = dcm_file[start_num:end_num]# create a small batch to work through and add to the master dataframe

for filename in dcm_file_subset['file_name']:
    # Crop the Image Base don .xml file 
    nii_im = sitk.ReadImage(filename)    
    nii_arr = sitk.GetArrayViewFromImage(nii_im)      
    
    try: 
        plt.figure()
        plt.imshow(nii_arr[0,], cmap=plt.cm.bone)
        plt.show()
        print("Enter y/n is image is a AP chest xray:")
        x=input()
        plt.close()  
    except:
        print('not an image')
        x = 'N/A'
    dcm_file.loc[dcm_file['file_name'] == filename, 'Chest'] = x

#Save the DF as a CSV
dcm_file.to_csv(os.path.join(SAVEPATH, 'AP_chest_radiographs_' + str(start_num) +\
                             '_' + str(end_num) + '.csv'), encoding='utf-8')