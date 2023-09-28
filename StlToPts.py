import open3d as o3d
from pyntcloud import PyntCloud
import pyvista as pv
input = 'D:/IACTA/new2/left_sub-0001.stl'

filename_stl = input
filename_ply = input.replace('.stl','.ply')
filename_pts = input.replace('.stl','.pts')

mesh = o3d.io.read_triangle_mesh(filename_stl)
o3d.io.write_triangle_mesh(filename_ply, mesh) #save as ply
anky = PyntCloud.from_file(filename_ply)
anky.points
number_of_points = 8000 #控制點的數量
anky_cloud = anky.get_sample("mesh_random", n=number_of_points, rgb=False, normals=True, as_PyntCloud=True)
filename_pts = filename_pts.replace('.pts' , '_' + str(number_of_points) + '.pts')
anky_cloud.to_file(filename_pts,sep=" ",header=0,index=0)