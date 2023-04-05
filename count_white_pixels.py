import numpy as np
import cv2
import pydicom as dicom
from PIL import Image

img  = cv2.imread('18.jpg')
#cv2.imshow('iamge',img)
#cv2.waitKey()
area = 0

def ostu(img):
    global area
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰度
    blur = cv2.GaussianBlur(img, (5,5),0,) #閥值設定0 高斯模糊
    cv2.imshow('iamge',img)
    cv2.waitKey()
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #二值化 0 = black ; 1 = white
    # cv2.imshow('image', th3)
    # a = cv2.waitKey(0)
    # print a
    height, width = th3.shape
    for i in range(height):
        for j in range(width):
            if th3[i, j] == 255:
                area += 1
    return area

ostu(img)
print(area)
