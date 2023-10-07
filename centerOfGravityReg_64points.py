#計算點雲檔案中27個重心點的位置
import numpy as np

#點雲檔案路徑
file = r"C:\ICP\1003\1003\right_sub-0001_8000.pts"
#file = r"C:\ICP\1003\1003\left_sub-0001_8000.pts"
# 輸出檔案路徑
output_file = r"C:\ICP\1003\1003\right_sub-0001_8000_64centerOfG.pts"
#output_file = r"C:\ICP\1003\1003\left_sub-0001_8000_64centerOfG.pts"

# 讀取點雲檔案
data = np.loadtxt(file)

# 切割
x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])

x = (
    round(x_min), 
    round(x_min + (x_max - x_min) * 1/4, 3),
    round(x_min + (x_max - x_min) * 2/4, 3),
    round(x_min + (x_max - x_min) * 3/4, 3),
    round(x_max, 3)
)
y = (
    round(y_min), 
    round(y_min + (y_max - y_min) * 1/4, 3),
    round(y_min + (y_max - y_min) * 2/4, 3),
    round(y_min + (y_max - y_min) * 3/4, 3),
    round(y_max, 3)
)
z = (
    round(z_min), 
    round(z_min + (z_max - z_min) * 1/4, 3),
    round(z_min + (z_max - z_min) * 2/4, 3),
    round(z_min + (z_max - z_min) * 3/4, 3),
    round(z_max, 3)
)


# 分割點雲資料
segments = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            # x方向的判斷
            if i == 3:  # 最後一個子立方體
                mask_x = (data[:, 0] >= x[i]) & (data[:, 0] <= x[i + 1])
            else:
                mask_x = (data[:, 0] >= x[i]) & (data[:, 0] < x[i + 1])

            # y方向的判斷
            if j == 3:
                mask_y = (data[:, 1] >= y[j]) & (data[:, 1] <= y[j + 1])
            else:
                mask_y = (data[:, 1] >= y[j]) & (data[:, 1] < y[j + 1])

            # z方向的判斷
            if k == 3:
                mask_z = (data[:, 2] >= z[k]) & (data[:, 2] <= z[k + 1])
            else:
                mask_z = (data[:, 2] >= z[k]) & (data[:, 2] < z[k + 1])

            # 結合三個方向的判斷
            mask = mask_x & mask_y & mask_z
            

            segment = data[mask]

            segments.append(segment)
            
# 計算重心點座標
centroids = []
for segment in segments:
    if len(segment) > 0:
        centroid = np.mean(segment, axis=0)
        centroids.append(centroid)
centroids = np.array(centroids)

# 儲存到新的pts檔案
np.savetxt(output_file, centroids, delimiter=' ', fmt='%.6f')


