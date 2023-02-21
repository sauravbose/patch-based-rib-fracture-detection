#!/home/boses1/miniconda3/bin/python3

"""
# AP Chest Bounding Box (xml--> csv)

Last Edited: 10/22/2021
Author: Saurav Bose, Daniella Patton

Description:
The AP chest bounding boxes were manually created using ybat and saved as xml
files. The x and y min and max values are saved in each file. This .py
file pulls this information, in addition the the actual size of the input
image and saves this as a .csv file.
"""
#%% PACKAGES
import os
import glob
import pandas as pd
import SimpleITK as sitk
import xml.etree.ElementTree as ET
import re
pd.options.mode.chained_assignment = None

#%% PATH
xml_dir = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/test_set/NoRibFractureData/xml'
img_dir = "/mnt/isilon/prarie/boses1/Projects/RibFracture/data/test_set/NoRibFractureData/nii"
savecsv_dir = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/metadata/test_set'
result_file_name = 'APChestBB_No_Fracture.csv'

#Change label to 0 for no rib fracture and to 1 for rib fracture
label = 0

#%% MAIN
def main():

    os.chdir(xml_dir)
    data = []
    for xml_file in [f for f in glob.glob("*.xml")]:
        image_name = xml_file[:-4] + '.nii'
        name = xml_file[:-4]

        try:
            nii_im = sitk.ReadImage(os.path.join(img_dir, image_name))
        except:
            print(f'Image {name} not found')
            continue

        nii_arr = sitk.GetArrayViewFromImage(nii_im)
        x_orig = nii_arr.shape[2]
        y_orig = nii_arr.shape[1]

        tree = ET.parse(xml_file)
        root = tree.getroot()
        as_string = ET.tostring(root, encoding='utf8').decode('utf8')
        xmin = re.search('<xmin>(.*)</xmin>', as_string).group(1)
        xmax = re.search('<xmax>(.*)</xmax>', as_string).group(1)
        ymin = re.search('<ymin>(.*)</ymin>', as_string).group(1)
        ymax = re.search('<ymax>(.*)</ymax>', as_string).group(1)
        data.append([label, image_name, name, xml_file,
                     int(x_orig), int(y_orig),
                     int(xmin), int(xmax), int(ymin), int(ymax),
                     int(xmax) - int(xmin), int(ymax) - int(ymin)])


    cols = ['label', 'image', 'name','xml',
            'x_orig', 'y_orig',
            'xmin', 'xmax', 'ymin' , 'ymax', 'x_crop', 'y_crop']
    df_to_save = pd.DataFrame(data, columns = cols)
    df_to_save.to_csv(os.path.join(savecsv_dir, result_file_name))

if __name__ == "__main__":
    main()
