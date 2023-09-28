import numpy as np
from scipy.spatial import KDTree

def load_pts_file(file_path):
    """從pts檔案中讀取點的座標數據"""
    points = np.loadtxt(file_path)
    return points

def write_pts_file(file_path, points):
    """將點的座標數據寫入到.pts檔案中"""
    np.savetxt(file_path, points)

def find_nearest_points(pts1, pts2):
    """找到第一個點集pts1中每個點對應到第二個點集pts2中距離最短的點"""
    kdtree = KDTree(pts2)
    distances, indices = kdtree.query(pts1)
    return distances, pts2[indices]

def main():
    # 讀取第一個pts檔案
    pts1_file = 'D:/IACTA/new/right_sub-0001_8000.pts'
    pts1 = load_pts_file(pts1_file)
    print(pts1)

    # 讀取第二個pts檔案
    pts2_file = 'D:/IACTA/new/left_sub-0001_8000.pts'
    pts2 = load_pts_file(pts2_file)

    # 找到每個pts1中的點對應到pts2中距離最短的點
    distances, nearest_points = find_nearest_points(pts1, pts2)

    # 創建一個新的pts1檔案來保存距離大於20的點座標
    dis = 6 #設定點距離相差多少以上要保留座標
    new_pts_file = "D:/IACTA/new2/" + str(dis) + "_distant_above_" + pts1_file.split("/")[-1] + ".pts"
    new_pts = pts1[distances > dis]

    # 寫入新的pts檔案
    write_pts_file(new_pts_file, new_pts)
    return 0

if __name__ == "__main__":
    main()
