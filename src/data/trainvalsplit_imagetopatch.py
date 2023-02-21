# -*- coding: utf-8 -*-
"""
# Training & Validation Split --> Patches

Last Edited: 8/9/2021
Author: Daniella Patton

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
import seaborn as sns
import glob

file_path = '/mnt/isilon/prarie/pattondm/RibFracture/src/data'
from func.helper import plot_images, calculate_area, final_csv_creater, norib_final_csv_creater

#%% PATHS

path = "/mnt/isilon/prarie/RibFractureStudy/data/processed/training/RibFractureData" 
noribfrac_path = "/mnt/isilon/prarie/RibFractureStudy/data/processed/training/NoRibFractureData"
save_path = "/mnt/isilon/prarie/RibFractureStudy/data/processed"
#%% MAIN FUNCTION
def main(path, noribfrac_path, save_path):
    rand = 1942      
    
    # Splitting into training validation and test data
    os.chdir(path)
    data = pd.read_csv('RibFracture_Cropped_Resized.csv')
    
    # Split the Training Rib Fracture Data
    # Set aside 20 % of image data for the validation data
    # (split by unique accession number)
    
    data['acc'] = data.image.str[:-10]
    X = data.acc.unique()       
    
    X_train, X_val, _, _ = train_test_split(X, [0] * len(X), test_size=0.20, 
                                                        random_state=rand)
    X_train_rib_fracture = data[data.acc.isin(X_train)]
    X_val_rib_fracture = data[data.acc.isin(X_val)]

    print('The training data has', len(X_train), 'Unique accession and ', 
          len(X_train_rib_fracture), 'fractures')
    print('The validation data has', len(X_val), 'Unique accession and ', 
          len(X_val_rib_fracture), 'fractures')
    
    # Creating Patches from the Original Dataset 
    training_patches = final_csv_creater(X_train_rib_fracture)
    validation_patches = final_csv_creater(X_val_rib_fracture)
    training_patches['dir'], validation_patches['dir'] = path, path
    training_patches['Rad Type'], validation_patches['Rad Type'] = 'Y', 'Y'
    
    # Creating No Rib Fracture 
    os.chdir(noribfrac_path)
    norib_list = [file for file in glob.glob("*.nii")]
    #print('Number of no rib images: ', len(norib_list))
    norib_df = pd.DataFrame(norib_list, columns = ['image'])
    norib_df['acc'] = norib_df.image.str[:-10]
    X = norib_df.acc.unique() 
    X_train, X_val, _, _ = train_test_split(X, [0] * len(X), test_size=0.20, 
                                                        random_state=rand)
    #print(len(X_train), len(X_val), len(X_train) +  len(X_val))
    X_train_norib_fracture = norib_df[norib_df.acc.isin(X_train)]
    X_val_norib_fracture = norib_df[norib_df.acc.isin(X_val)]
    training_patches_norib = norib_final_csv_creater(X_train_norib_fracture)
    validation_patches_norib = norib_final_csv_creater(X_val_norib_fracture)
    
    training_patches_norib['dir'], validation_patches_norib['dir'] = noribfrac_path, noribfrac_path
    training_patches_norib['Rad Type'], validation_patches_norib['Rad Type'] = 'N', 'N'

    # Concatentaing the training rib fracture and no rib fracture df to make
    # our final test set
    training_patches_f = pd.concat([training_patches, training_patches_norib])
    validation_patches_f = pd.concat([validation_patches, validation_patches_norib])
    
    # Save the training and validation dataset as csv
    training_patches_f.to_csv(os.path.join(save_path, 'training.csv'))
    validation_patches_f.to_csv(os.path.join(save_path, 'validation.csv'))  
    
    # Plotting the Distirbution of the data
    int2id = {0:'No Fracture', 
              1:'<= 60% Area or Uncertain', 
              2:'Fracture'}
    
    # Plotting the Training Data  (without no rib fracture data)
    print('WITHOUT NO RIB FRACTURE DATA \n')
    plt.figure(0)
    training_patches['labels'] = training_patches['label'].map(int2id)
    sns.set_theme(style="darkgrid")
    ax1 = sns.countplot(x="labels", data=training_patches)
    ax1.set_title('Training Data Patches Distibution (Rib Fracture Only)')
    print('Training Data Patches Distibution (Rib Fracture Only)')
    print(training_patches.labels.value_counts(normalize=True))

    print('\n')
    # Plotting the Valdiation Data (without no rib fracture data)
    plt.figure(1)    
    validation_patches['labels'] = validation_patches['label'].map(int2id)
    sns.set_theme(style="darkgrid")
    ax2 = sns.countplot(x="labels", data=training_patches)
    ax2.set_title('Validation Data Patches Distibution (Rib Fracture Only)')
    print('Validation Data Patches Distibution (Rib Fracture Only)')
    print(validation_patches.labels.value_counts(normalize=True))
     
    # Plotting the Training Data  (without no rib fracture data)
    plt.figure(2)
    training_patches_f['labels'] = training_patches_f['label'].map(int2id)
    sns.set_theme(style="darkgrid")
    ax3 = sns.countplot(x="labels", data=training_patches_f)
    ax3.set_title('Training Data Patches Distibution')
    print('Training Data Patches Distibution')    
    print(training_patches_f.labels.value_counts(normalize=True))
    
    print('\n')
    # Plotting the Valdiation Data (without no rib fracture data)
    plt.figure(3)
    validation_patches_f['labels'] = validation_patches_f['label'].map(int2id)
    sns.set_theme(style="darkgrid")
    ax4 = sns.countplot(x="labels", data=validation_patches_f)
    ax4.set_title('Validation Data Patches Distibution')
    print('Validation Data Patches Distibution')
    print(validation_patches_f.labels.value_counts(normalize=True))
   
if __name__ == "__main__":
    main(path, noribfrac_path, save_path) 
  
    
    
    





