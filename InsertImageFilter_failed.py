#感覺程式邏輯是對的但是InsertImageFilter函式不能用
import SimpleITK as sitk
import pydicom
import os
import numpy as np

# Set the path to the input DICOM files directory
dicom_dir = 'C:/meeting/2D/test/dicomWithSilceLocation/'

# Load the DICOM files and sort them by SliceLocation
dicom_files = []
for file_name in os.listdir(dicom_dir):
    dicom_file = pydicom.dcmread(os.path.join(dicom_dir, file_name))
    dicom_files.append(dicom_file)
dicom_files.sort(key=lambda x: x.SliceLocation)

# Get the pixel spacing and slice spacing from the first DICOM file
pixel_spacing = dicom_files[0].PixelSpacing
slice_spacing = dicom_files[0].SliceThickness

# Calculate the output image size
width = dicom_files[0].Columns
height = dicom_files[0].Rows
depth = len(dicom_files)

print('pixel_spacing',pixel_spacing)
print('slice_spacing',slice_spacing)
print('width',width)
print('height',height)
print('depth',depth)

# Create an empty image with the desired size and spacing
image = sitk.Image(width, height, depth, sitk.sitkInt16)
image.SetSpacing((pixel_spacing[0], pixel_spacing[1], slice_spacing))
print(image)
# Copy the pixel data from the DICOM files into the image
for i in range(depth):
    image_array = dicom_files[i].pixel_array
    image_array = sitk.GetImageFromArray(image_array)
    image_array.SetSpacing((pixel_spacing[0], pixel_spacing[1]))
    image_array = sitk.Cast(image_array, sitk.sitkInt16)
    mage = sitk.InsertImageFilter().Execute(image, image_array, [0, 0, i])


# Save the image as a NIfTI file
output_file = "C:/meeting/2D/test/dicomBacktoNifti/test25.nii.gz"
sitk.WriteImage(image,output_file)
