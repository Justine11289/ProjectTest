#pip install nibabel
from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib
import open3d as o3d
from pyntcloud import PyntCloud
import pyvista as pv

#pip install voxelfuse
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import generateMaterials
#import nibabel



# matplotlib.use('TkAgg')
# 文件名，nii或nii.gz
gz_filename = 'amos_0004'
example_filename = 'label_Tr\\' + gz_filename + '.nii.gz'
img = nib.load(example_filename)

#----------------------------------------------------------------------------
#Load nifti mask and convert to numpy
#mask_dir = '' #Put the directory of the mask file here
#mask_nifti = nibabel.load(mask_dir)
#mask_nifti = nib.load(mask_dir)
mask_nifti = img
mask_npy = mask_nifti.get_fdata()

#Convert all nonzero labels (i.e. lesion labels) to 1
mask_npy[mask_npy != 0] = 1

#Convert to mesh and save 
model = VoxelModel(mask_npy, generateMaterials(1))
mesh = Mesh.fromVoxelModel(model)
export_stl_name = gz_filename + '.stl'
mesh.export(export_stl_name)
#------------------------------------------------------------------------------