import pylab
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydicom







ImageFile = pydicom.dcmread('C:/meeting/2D/test/test/S4-A4-0018.dcm')
image_array = ImageFile.pixel_array
high = np.max(image_array) # 找到最大的
low = np.min(image_array)# 找到最小的


lungwin = np.array([low * 1., high * 1.]) # 将pydicom解析的像素值转换为array
newimg = (image_array - lungwin[0]) / (lungwin[1] - lungwin[0]) # 将像素值归一化0-1
newimg = (newimg * 255).astype('uint8') # 再转换至0-255，且将编码方式由原来的unit16转换为unit8
file = 'C:/meeting/2D/test/dicomToTiff_18.tiff'
cv2.imwrite(file,newimg)