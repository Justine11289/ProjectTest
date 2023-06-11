import numpy as np

def point_cloud_translation(point_cloud, translation_units):
    """
    對點雲進行平移擴增

    參數：
    point_cloud：ndarray，形狀為(N, 3)，表示包含N個點的點雲
    translation_units：list，形狀為(3,)，表示每個坐標軸上的平移單位數

    返回值：
    translated_point_cloud：ndarray，形狀為(N, 3)，表示平移後的點雲
    """

    translation = np.array(translation_units)
    translated_point_cloud = point_cloud + translation

    return translated_point_cloud


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

# 定義平移範圍，這裡設置為在每個坐標軸上平移n個單位
translation_units = [10, 10, 10]

# 呼叫平移擴增函式
translated_point_cloud = point_cloud_translation(point_cloud, translation_units)

# 儲存平移後的點雲
output_file_path = r'C:/Users/User/Desktop/專題/pts/a/translated_point_cloud152.txt'
save_point_cloud(translated_point_cloud, output_file_path)

print("平移後的點雲已儲存至檔案:", output_file_path)
