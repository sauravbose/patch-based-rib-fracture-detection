#!/home/boses1/miniconda3/bin/python3

import shutil, os
import pandas as pd

# metadata_path = '../data/metadata/validation_112_CORRECTED.csv'
metadata_path =  '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/metadata/'
img_source_dir = '/mnt/isilon/prarie/RibFractureStudy/data/processed/train_val/data/'
img_destination_dir = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/processed_sample/validation/'

validation_df = pd.read_csv(metadata_path + 'validation_112_CORRECTED.csv')
validation_df = validation_df.loc[validation_df.label!=1].reset_index(drop=True)
validation_df["label"].replace({2: 1}, inplace=True)

files = validation_df.image.unique()

for f in files:
    shutil.copy(img_source_dir + f, img_destination_dir)
