#可以依照DICOM原先尺寸疊圖成NII
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

# Create an empty numpy array with the desired size
image_array = np.zeros((height, width, depth), dtype=np.int16)

# Copy the pixel data from the DICOM files into the numpy array
for i, dicom_file in enumerate(dicom_files):
    image_array[:, :, i] = dicom_file.pixel_array

#print(image_array)

# Convert the numpy array to a SimpleITK image
image = sitk.GetImageFromArray(image_array)

# Set the image spacing
#image.SetSpacing((pixel_spacing[0], pixel_spacing[1], slice_spacing))
image.SetSpacing((slice_spacing,pixel_spacing[0], pixel_spacing[1]))
print(image)

output_file = "C:/meeting/2D/test/dicomBacktoNifti/test27.nii.gz"
sitk.WriteImage(image,output_file)