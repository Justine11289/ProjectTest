import cv2
import os
import pydicom

path = 'C:/meeting/2D/test/dicomWithSilceLocation/test3_dicom/'
out_path = 'C:/meeting/2D/test/dicomWithSilceLocation/test3_png/'
test_list = [ f for f in  os.listdir(path)]

for f in test_list:
    ds = pydicom.read_file(path + f) # read dicom 
    pname = path.split("/")[-2]# 先用split方法切割路徑，[-1]代表取最後一個元素，也就是檔案名稱
    pname = pname[:-11] # 再用切片方法切掉檔案名稱中的".nii.gz"字串
    img = ds.pixel_array # get image array
    cv2.imwrite(out_path + pname + f.replace('.dcm','_mask.png'),img)