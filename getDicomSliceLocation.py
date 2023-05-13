import numpy as np
import pydicom
import cv2
import os
import matplotlib.pyplot as plt

#得到資料夾中所有dicom檔案的silce location並且當成檔案名稱另存新檔
def get_dicom_slice_location(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            continue
        ds = pydicom.dcmread(file_path)
        img_array = ds.pixel_array
        high = np.max(img_array)
        low = np.min(img_array)
        lungwin = np.array([low * 1., high * 1.])
        newimg = (img_array - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg = (newimg * 255).astype('uint8')
        sliceLocation = ds.SliceLocation
        sliceLocation = str(sliceLocation)
        #print(sliceLocation) 確認資訊用

        file = 'C:/meeting/2D/test/dicomWithSilceLocation/test3_dicom/'
        file += sliceLocation
        file += '.dcm'
        ds.save_as(file,write_like_original=True)


directory = 'C:/meeting/2D/test/test/'
get_dicom_slice_location(directory)

