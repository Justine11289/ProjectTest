#切割大腦左右各1/2並分別保存檔案
import nibabel as nib
import numpy as np

nifti_file = 'D:/IACTA/new/sub-0001-1.nii.gz'
output_file_left = 'D:/IACTA/new/' + 'left_' + nifti_file.split('/')[-1]
output_file_right = 'D:/IACTA/new/' + 'right_' + nifti_file.split('/')[-1]

# 读取NIfTI文件
nifti_data = nib.load(nifti_file)
nifti_header = nifti_data.header
nifti_affine = nifti_data.affine

# 获取NIfTI数据和形状
data = nifti_data.get_fdata()
shape = data.shape

#右半邊
# 计算切割位置
cut_index_left = int(shape[0] * (1/2))

# 切割数据
sliced_data = data[:cut_index_left, :,:]

# 更新NIfTI头信息的维度
nifti_header.set_data_shape(sliced_data.shape)

# 保存切割后的NIfTI文件
sliced_nifti = nib.Nifti1Image(sliced_data, nifti_affine, header=nifti_header)
nib.save(sliced_nifti, output_file_right)


#左半邊
nifti_data = nib.load(nifti_file)
nifti_header = nifti_data.header
nifti_affine = nifti_data.affine

# 获取NIfTI数据和形状
data = nifti_data.get_fdata()
shape = data.shape
# 计算切割位置
cut_index_right = int(shape[0] * (1/2))

# 切割数据
sliced_data = data[cut_index_right:, :,:]

# 更新NIfTI头信息的维度
nifti_header.set_data_shape(sliced_data.shape)

# 保存切割后的NIfTI文件
sliced_nifti = nib.Nifti1Image(sliced_data, nifti_affine, header=nifti_header)
nib.save(sliced_nifti, output_file_left)

