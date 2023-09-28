import nibabel as nib
import numpy as np

def mirror_nifti(nifti_file,output_file):
    # 讀取NIfTI檔案
    nifti = nib.load(nifti_file)
    data = nifti.get_fdata()
    
    # 進行左右鏡射
    mirrored_data = np.flip(data, axis=0)
    
    # 更新NIfTI物件的數據
    mirrored_nifti = nib.Nifti1Image(mirrored_data, nifti.affine)
    
    # 儲存鏡射後的NIfTI檔案
    nib.save(mirrored_nifti, output_file)

# 呼叫函式並傳入NIfTI檔案的路徑
input_file = 'C:/meeting/3D/halfBrain/half_sub-0222.nii.gz'
output_file = 'C:/meeting/3D/halfBrain/half_sub-0222_mirrored.nii.gz'
mirror_nifti(input_file,output_file)
