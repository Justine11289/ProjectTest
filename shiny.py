import os
import cv2
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from stl import mesh
import open3d as o3d
from scipy.spatial import KDTree
from pyntcloud import PyntCloud
from nibabel.testing import data_path

def nifti_to_pts(nifti_file, output_dir):
    # Get NIfTI data
    data = nifti_file.get_fdata()
    points = np.argwhere(data != 0)
    pts_file = os.path.join(output_dir, nifti_file.get_filename().replace(".nii.gz", ".pts"))
    np.savetxt(pts_file, points, delimiter=' ', fmt='%f')

def process(input_file, output_directory):
    # Load NIfTI images
    nii_img = nib.load(input_file)
    nii_img_data = nii_img.get_fdata()
    img_shape = nii_img_data.shape
    # Set NumPy output format options
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    # Remove head shell
    slices = []
    for index in range(img_shape[2]):
        slice = nii_img_data[:, :, index]
        vessel = np.where(slice > 3000, slice, 0)
        slices.append(vessel)

    # New NIfTI images
    slices = np.array(slices)
    affine_matrix = nii_img.affine
    affine_matrix[0][0] = -affine_matrix[0][0]
    affine_matrix = affine_matrix[[1, 2, 0, 3]]
    new_nii_img = nib.Nifti1Image(slices, affine_matrix)
    output_file = os.path.join(output_directory, input_file.split("/")[-1])
    nib.save(new_nii_img, output_file)
    nifti_to_pts(new_nii_img, output_directory)

def main():
    input_dir = 'D:/Project/Shiny_icarus/'
    output_dir = 'D:/Project/Shiny_icarus/new/'

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all .nii.gz files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            input_file = os.path.join(input_dir, filename)
            process(input_file, output_dir)

if __name__ == '__main__':
    main()
