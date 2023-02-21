#!/home/boses1/miniconda3/bin/python3

import os
import glob

file_path = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/test_set/RibFractureData/final_test_set_10_20_21/segmentations_renamed/'

os.chdir(file_path)

for segmentation in [f for f in glob.glob("*.nrrd")]:
    new_file_name = segmentation.split('.')[0] + '-label.nrrd'
    os.rename(file_path+segmentation, file_path+new_file_name)
