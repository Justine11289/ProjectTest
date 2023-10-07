#計算點雲檔案中27個重心點的位置
import numpy as np

#點雲檔案路徑
file = 'C:/ICP/new/new/mirror_left_sub-0001_8000.pts'
output_file = 'C:/ICP/new/new/right_sub-0001_8000_centerOfG.pts'
# 讀取點雲檔案
data = np.loadtxt(file)

# 切割成八個小部分
x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])

x,y,z = [5],[5],[5]
x = round(x_min) , round((x_max - x_min) * 1/3 , 3) , round((x_max - x_min) * 2/3 , 3) , round((x_max - x_min) * 3/3 , 3) , round(x_max)
y = round(y_min) , round((y_max - y_min) * 1/3 , 3) , round((y_max - y_min) * 2/3 , 3) , round((y_max - y_min) * 3/3 , 3) , round(y_max)
z = round(z_min) , round((z_max - z_min) * 1/3 , 3) , round((z_max - z_min) * 2/3 , 3) , round((z_max - z_min) * 3/3 , 3) , round(z_max)

# 分割點雲資料
segments = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            mask = (
                (data[:, 0] >= x[i] ) & (data[:, 0] < x[i + 1] ) &
                (data[:, 1] >= y[j] ) & (data[:, 1] < y[j + 1] ) &
                (data[:, 2] >= z[k] ) & (data[:, 2] < z[k + 1] )
            )
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


