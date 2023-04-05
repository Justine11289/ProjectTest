import pydicom
import matplotlib
import matplotlib.pyplot as plt
from numba import jit 
import numpy as np
import os


df = pydicom.dcmread('C:/meeting/2D/test/test/S4-A4-0018.dcm')
plt.figure(figsize=(6, 6))

#打印患者信息部分
print("圖像行數：{}".format(df.Rows))
print("圖像列數：{}".format(df.Columns))
print("切片厚度(mm)：{}".format(df.SliceThickness))
print("圖像像素間距(mm)：{}".format(df.PixelSpacing))
print("窗位：{}".format(df.WindowCenter))
print("窗寬：{}".format(df.WindowWidth))
print("截取(轉換CT值)：{}".format(df.RescaleIntercept))
print("斜率(轉換CT值)：{}".format(df.RescaleSlope))
print("其他信息：")
print(df.data_element)

#获取图像部分
img = df.pixel_array
plt.imshow(img, 'gray')
print("图像形状:{}".format(img.shape))
plt.show()

#加载这个序列
def load(path):
    #加载这个系列切片
    slices = [pydicom.read_file(os.path.join(path,s), force=True) for s in os.listdir(path)]
    #按z轴排序切片
    slices.sort(key= lambda x: float(x.ImagePositionPatient[2]))
    #计算切片厚度
    slice_thick = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    #同一厚度
    for s in slices:
        s.SliceThickness  = slice_thick
    return slices

#调整窗宽窗位之前把图像像素值变灰CT值
def get_pixel_hu(slices):
    #CT值和灰度值的转换公式 Hu = pixel * slope + intercept
    #三维数据 z,y,x排列
    images = np.stack([s.pixel_array  for s in slices])
    # 转换为int16
    images = images.astype(np.int16)
    # 设置边界外的元素为0
    images[images == -2048] = 0
    # 转换为HU单位
    for slice_num in range(len(slices)):
        #获取截取
        intercept = slices[slice_num].RescaleIntercept
        #获取斜率
        slope = slices[slice_num].RescaleSlope
        #有的图像就已经是CT值（HU值），这时候读出来的Solpe=1，Intercept=0。
        if slope != 1:
            images[slice_num] = slope * images[slice_num].astype(np.float64)
            images[slice_num] = images[slice_num].astype(np.int16)
        images[slice_num] = images[slice_num] + np.int16(intercept)
    return images


@jit(nopython=True)
def calc(img_temp, rows, cols, minval,maxval):
    for i in np.arange(rows):
        for j in np.arange(cols):
            #避免除以0的报错
            if maxval - minval == 0:
                result = 1
            else:
                result = maxval - minval
            img_temp[i, j] = int((img_temp[i, j] - minval) / result * 255)

# 調整CT圖像的窗寬窗位
def setDicomWinWidthWinCenter(img_data,winwidth, wincenter):
    minval = (2*wincenter - winwidth) / 2.0 + 0.5
    maxval = (2*wincenter + winwidth) / 2.0 + 0.5
    for index in range(len(img_data)):
        img_temp = img_data[index]
        rows, cols = img_temp.shape
        calc(img_temp, rows, cols, minval, maxval)
        img_temp[img_temp < 0] = 0
        img_temp[img_temp > 255] = 255
        img_data[index] = img_temp
    return img_data


#設置窗寬窗位
winwidth = 30
wincenter = 160
#讀取整个系列dicom文件
path = 'C:/meeting/2D/test/test/'
patient = load(path)
#像素值转成CT值
patient_pixels = get_pixel_hu(patient)
#改变窗宽窗位
patient_pixels = setDicomWinWidthWinCenter(patient_pixels,winwidth,wincenter)
plt.figure(figsize=(6, 6))
plt.imshow(patient_pixels[30], 'gray')
plt.show()
plt.imsave('C:/meeting/2D/test/'+'18.jpg',patient_pixels[30]/255.0)