# pip install open3d
# pip install pyntcloud
# pip install pyvista
import open3d as o3d
from pyntcloud import PyntCloud
import pyvista as pv


filename = 'Untitled00014'
filename_stl = filename + '.stl'
filename_ply = filename + '.ply'
filename_pts = filename + '.pts'
filename_txt = filename + '.txt'
mesh = o3d.io.read_triangle_mesh(filename_stl)
o3d.io.write_triangle_mesh(filename_ply, mesh)
anky = PyntCloud.from_file(filename_ply)
anky.points
print(anky.points)
anky_cloud = anky.get_sample("mesh_random", n=80000, rgb=False, normals=True, as_PyntCloud=True)#n代表點的數量
anky_cloud.to_file(filename_pts,sep=" ",header=0,index=0)#把座標寫到pts檔案裡
print(anky.points.describe())
max_z = max(anky.points["z"])
min_z = min(anky.points["z"])
max_x = max(anky.points["x"])
min_x = min(anky.points["x"])

print("\n")
print("max_z ")
print(max_z)
print("min_z ")
print(min_z)
print("max_x ")
print(max_x)
print("min_x ")
print(min_x)
print("\n")

max_z_locate = anky.points.z == max_z
print(max_z_locate)
#anky_cloud = anky.get_sample("mesh_random", n=80000, rgb=False, normals=True, as_PyntCloud=True)#n代表點的數量
#anky_cloud.to_file(filename_txt,sep=" ",header=0,index=0)#把座標寫到txt檔案裡
max_z_locate.to_file("locate.txt")#把座標寫到txt檔案裡