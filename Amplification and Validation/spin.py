import numpy as np
from scipy.spatial.transform import Rotation


def point_cloud_rotation(point_cloud, rotation_angle_deg):
    """
    對點雲進行旋轉擴增

    參數：
    point_cloud：ndarray，形狀為(N, 3)，表示包含N個點的點雲
    rotation_angle_deg：float，表示旋轉角度（單位：度）

    返回值：
    rotated_point_cloud：ndarray，形狀為(N, 3)，表示旋轉後的點雲
    """

    # 將旋轉角度轉換為弧度
    rotation_angle_rad = np.radians(rotation_angle_deg)

    # 定義旋轉軸為z軸
    rotation_axis = np.array([0, 0, 1])

    # 創建旋轉物件
    rotation = Rotation.from_rotvec(rotation_angle_rad * rotation_axis)

    # 對點雲進行旋轉
    rotated_point_cloud = rotation.apply(point_cloud)

    return rotated_point_cloud


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


def save_point_cloud(point_cloud, file_path):
    """
    儲存點雲資料到檔案

    參數：
    point_cloud：ndarray，形狀為(N, 3)，表示包含N個點的點雲
    file_path：str，點雲檔案的路徑
    """

    with open(file_path, 'w') as file:
        for i in range(point_cloud.shape[0]):
            file.write(f"{point_cloud[i, 0]} {point_cloud[i, 1]} {point_cloud[i, 2]}\n")


# 讀取點雲資料
file_path = r'C:/Users/User/Desktop/專題/pts/a/152_0000-Cloud.txt'
point_cloud = read_point_cloud(file_path)

# 定義旋轉角度（單位：度）
rotation_angle_deg = 45

# 呼叫旋轉擴增函式
rotated_point_cloud = point_cloud_rotation(point_cloud, rotation_angle_deg)

# 儲存旋轉後的點雲
output_file_path = r'C:/Users/User/Desktop/專題/pts/a/rotated_point_cloud152.txt'
save_point_cloud(rotated_point_cloud, output_file_path)

print("旋轉後的點雲已儲存至檔案:", output_file_path)
