import numpy as np

def compare_point_clouds(point_cloud1, point_cloud2, tolerance):
    """
    比對兩個點雲資料的一致性

    參數：
    point_cloud1：ndarray，形狀為(N, 3)，表示第一個點雲資料
    point_cloud2：ndarray，形狀為(M, 3)，表示第二個點雲資料
    tolerance：float，比對容忍的誤差範圍

    返回值：
    is_consistent：bool，指示兩個點雲資料是否一致
    """

    if point_cloud1.shape != point_cloud2.shape:
        return False

    num_points = point_cloud1.shape[0]
    for i in range(num_points):
        distance = np.linalg.norm(point_cloud1[i] - point_cloud2[i])
        if distance > tolerance:
            return False

    return True



def read_point_cloud(file_path):
    """
    從檔案讀取點雲資料

    參數：
    file_path：str，點雲檔案的路徑

    返回值：
    point_cloud：ndarray，形狀為(N, 3)，表示包含N個點的點雲
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    point_cloud = []
    for line in lines:
        line = line.strip()
        if line:
            coords = line.split(' ')
            point_cloud.append([float(coords[0]), float(coords[1]), float(coords[2])])

    point_cloud = np.array(point_cloud)

    return point_cloud


# 讀取兩個點雲資料
file_path1 = r'C:/Users/User/Desktop/專題/pts/a/translated_point_cloud152.txt'
file_path2 = r'C:/Users/User/Desktop/專題/pts/a/152_0000-Cloud.txt'
point_cloud1 = read_point_cloud(file_path1)
point_cloud2 = read_point_cloud(file_path2)

# 比對點雲資料的一致性
tolerance = 0  # 定義容忍的誤差範圍
is_consistent = compare_point_clouds(point_cloud1, point_cloud2, tolerance)

if is_consistent:
    print("兩個點雲資料一致")
else:
    print("兩個點雲資料不一致")