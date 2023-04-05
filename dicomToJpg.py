import pydicom # 用来解析dicom格式圖像的像素值
import numpy as np
import cv2 # 用於保存圖片
import os

# 定義dicom to jpg轉換函數
def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    """

    :param img: dicom图像的像素值信息
    :param low_window: dicom图像像素值的最低值
    :param high_window: dicom图像像素值的最高值
    :param save_path: 新生成的jpg图片的保存路径
    :return:
    """
    lungwin = np.array([low_window * 1., high_window * 1.]) # 將pydicom解析的像素值轉換為array
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0]) # 將像素值歸一化0-1
    newimg = (newimg * 255).astype('uint8') # 再转换至0-255，且將编碼方式由原来的unit16轉換為unit8
    # 用cv2寫入圖像指令，保存jpg即可
    file = 'testJPG.jpg'
    cv2.imwrite(file,newimg)

count = 187 # 设置了一个变量用来作为保存后jpg图像的名称的，可自行修改其他的
path = 'C:/meeting/2D/test' # dicom文件夹路径
filename = os.listdir(path) # 打开文件夹中的图像的文件名，作为列表返回
# print(filename) # 可查看一下文件夹下有哪些文件


ds = pydicom.dcmread('S4-A4-0018.dcm') # 解析一张dicom图片
img_array = ds.pixel_array # 将像素值信息提取
# ds_array = sitk.ReadImage(document)
# img_array = sitk.GetArrayFromImage(ds_array)
# shape = img_array.shape  # name.shape
# img_array = np.reshape(img_array, (shape[1], shape[2]))
high = np.max(img_array) # 找到最大的
low = np.min(img_array)# 找到最小的
# 调用函数，开始转换
outputpath = 'C:/meeting/2D/test'
convert_from_dicom_to_jpg(img_array, low, high, outputpath)
