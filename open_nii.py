#
from nibabel.viewers import OrthoSlicer3D 
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib
 

# matplotlib.use('TkAgg')
# 文件名，nii或nii.gz
gz_filename = 'amos_0004.nii.gz'
example_filename = 'label_Tr\\' + gz_filename
img = nib.load(example_filename)


# 打印文件信息
print(img)
print(img.dataobj.shape)
 
#shape不一定只有三个参数，打印出来看一下
width, height, queue = img.dataobj.shape
 
 
# 显示3D图像
OrthoSlicer3D(img.dataobj).show()
 
 
# 计算看需要多少个位置来放切片图
x = int((queue/10) ** 0.5) + 1
num = 1
# 按照10的步长，切片，显示2D图像
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(x, x, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
 
 
plt.show()