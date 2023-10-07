#可以做icp校正但是結果不準
import numpy as np
from scipy.spatial import KDTree

# 读取PTS文件并将点云数据存储为NumPy数组
def read_pts_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        coordinates = line.strip().split()
        if len(coordinates) == 3:
            point = [float(coordinates[0]), float(coordinates[1]), float(coordinates[2])]
            points.append(point)
    return np.array(points)

# 计算最佳的刚体变换矩阵
def icp_transform(source_points, target_points):
    source_tree = KDTree(source_points)
    _, indices = source_tree.query(target_points)
    correspondences = source_points[indices]
    centroid_source = np.mean(correspondences, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_centered = correspondences - centroid_source
    target_centered = target_points - centroid_target
    matrix = np.dot(np.transpose(source_centered), target_centered)
    U, _, V = np.linalg.svd(matrix)
    rotation_matrix = np.dot(V, np.transpose(U))

    # 反转旋转矩阵
    rotation_matrix[1, :] = -rotation_matrix[1, :]

    translation_vector = centroid_target - np.dot(rotation_matrix, centroid_source)
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

# 主函数
def main():
    source = 'C:/ICP/new/new/mirror_left_sub-0001_8000_centerOfG.pts'
    # 要進行校正的只有重心的pts
    target = 'C:/meeting/3D/centerOfGravity/0527testCOG_27points.pts'
    # 參考目標的只有重心的pts
    pts_file_path = 'C:/ICP/new/new/mirror_left_sub-0001_8000.pts'
    new_pts_file_path = 'C:/ICP/new/new/mirror_left_sub-0001_8000_ICP.pts'

    # 讀文件
    source_points = read_pts_file(source)
    target_points = read_pts_file(target)

    # 做27個點的ICP校正取得轉移矩陣
    transformation_matrix = icp_transform(source_points, target_points)
    print('矩陣:')
    print(transformation_matrix)


    #對目標半腦pts檔案套用矩陣
    pts = read_pts_file(pts_file_path)
    new_pts = np.dot(np.hstack((pts, np.ones((pts.shape[0], 1)))), transformation_matrix.T)
    new_pts = new_pts[:, :3]
    np.savetxt(new_pts_file_path, new_pts, delimiter=' ', fmt='%.6f')

if __name__ == '__main__':
    main()
