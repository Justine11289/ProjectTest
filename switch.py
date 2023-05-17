# pip install open3d
# pip install pyntcloud
# pip install pyvista
import copy
import open3d as o3d
from pyntcloud import PyntCloud
import pyvista as pv
import numpy as np
# mesh -> ply -> pts
mesh = o3d.io.read_triangle_mesh("Untitled00001.stl")
o3d.io.write_triangle_mesh("Untitled00001.ply", mesh)
anky = PyntCloud.from_file("Untitled00001.ply")
anky.points
anky_cloud = anky.get_sample("mesh_random", n=8000, rgb=False, normals=True, as_PyntCloud=True)
anky_cloud.to_file("Untitled00001.pts",sep=" ",header=0,index=0)

# ply -> pcd
ply = o3d.io.read_point_cloud("Untitled00001.ply")
o3d.io.write_point_cloud("Untitled00001.pcd", ply)
pcd = o3d.io.read_point_cloud("Untitled00001.pcd")
pcd.paint_uniform_color([1,0,0])
print("pcd質心:",pcd.get_center())

pcd_EulerAngle = copy.deepcopy(pcd)
R1 = pcd.get_rotation_matrix_from_xyz((0,np.pi/2,0))
pcd_EulerAngle.rotate(R1,center=(0,0,0))
pcd_EulerAngle.paint_uniform_color([0,0,1])
print("pcd質心:",pcd_EulerAngle.get_center())
o3d.io.write_point_cloud("Untitled00001.pcd", pcd_EulerAngle)