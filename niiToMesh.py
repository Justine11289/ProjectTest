import numpy as np
import nibabel as nib
from skimage import measure
from stl import mesh

def nifti_to_stl(nifti_file, stl_file):
    # 讀取NIfTI文件
    nifti_image = nib.load(nifti_file)
    nifti_data = nifti_image.get_fdata()

    # 將NIfTI數據轉換為二進制圖像（0表示空氣，非0表示實體）
    binary_image = np.where(nifti_data != 0, 1, 0).astype(np.uint8)

    # 使用Marching Cubes算法提取STL三角網格
    verts, faces, _, _ = measure.marching_cubes(binary_image)

    # 創建STL物體
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[face[j]]

    # 寫入STL文件
    stl_mesh.save(stl_file)

# 輸入NIfTI和STL文件的路徑
nifti_file = 'D:/IACTA/new2/right_sub-0001.nii.gz'
stl_file = 'D:/IACTA/new2/' + nifti_file.split('/')[-1].replace(".nii.gz",".stl")

# 呼叫轉換函數
nifti_to_stl(nifti_file, stl_file)
