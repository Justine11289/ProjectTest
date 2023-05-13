#可以把圖片疊起來但是三維空間沒有調整過間距，只有單純疊起來
import numpy as np
import pydicom
import cv2
import os
import matplotlib.pyplot as plt
import nibabel as nib

#宣告新的排序順序函數
def get_sort(list):
    num_str = ''.join(filter(str.isdigit, list))
    return int(num_str)

#讀取資料夾中dicom檔案
def read_dicom_series(directory):
    slices = []
    for file_name in sorted(os.listdir(directory),key = get_sort):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            continue
        ds = pydicom.dcmread(file_path)
        slices.append(ds.pixel_array)
        
    slices = np.stack(slices)
    return slices

#疊圖成nii
def create_nifti_file(data, filename):
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)

#設定路徑
data_dir = "C:/meeting/2D/test/dicomWithSilceLocation/"
data_dir = "C:/meeting/2D/test/123/"
output_file = "C:/meeting/2D/test/dicomBacktoNifti/test2.nii.gz"


#執行
data = read_dicom_series(data_dir)
#print(data)
print(data.shape)
# create_nifti_file(data, output_file)

