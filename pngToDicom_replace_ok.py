import numpy as np
import pydicom
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

#得到資料夾中所有dicom檔案的silce location並且當成檔案名稱另存新檔

def get_png_data(path,file_name):
    #print('check: ' + file_name) 確認資訊用
    test_list = os.listdir(path)
    for file in test_list:
        png_file_name = file.replace('_mask.png','.dcm')
        #print(png_file_name) 確認資訊用
        if png_file_name == file_name :
            png_image = Image.open(path + file)
            # 設定DICOM檔案的像素數據
            png_data = png_image.getdata()
            return png_data

def get_dicom_slice_location(directory,png_directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            continue
        ds = pydicom.dcmread(file_path)
        img_array = ds.pixel_array
        img_array = get_png_data(png_directory,file_name)


        high = np.max(img_array)
        low = np.min(img_array)
        lungwin = np.array([low * 1., high * 1.])
        newimg = (img_array - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg = (newimg * 255).astype('uint8')
        sliceLocation = ds.SliceLocation
        sliceLocation = str(sliceLocation)
        #print(sliceLocation) 確認資訊用

        file = 'C:/meeting/2D/test/dicomWithSilceLocation/test3_pngBackToDicom/'
        file += sliceLocation
        file += '.dcm'
        ds.save_as(file,write_like_original=True)





directory = 'C:/meeting/2D/test/dicomWithSilceLocation/test3_dicom/'
png_directory = 'C:/meeting/2D/test/dicomWithSilceLocation/test3_png/'
get_dicom_slice_location(directory,png_directory)
