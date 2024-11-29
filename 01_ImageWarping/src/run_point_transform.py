import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-6):
    warped_image = np.array(image)
    n=len(source_pts)
    bag_ori=np.array((warped_image.shape[0],warped_image.shape[1],n))

    h, w = warped_image.shape[:2]  # 获取图像高度和宽度
    n = len(source_pts)  # 控制点数量

    # 创建坐标网格
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))

    # 初始化 bag_ori 数组
    bag_ori = np.zeros((h, w, n), dtype=np.float32)

    # 将控制点和网格坐标进行广播
    for i in range(n):
        # 计算每个像素到控制点 i 的距离
        dx = x_indices - source_pts[i][0]  # 横坐标差
        dy = y_indices - source_pts[i][1]  # 纵坐标差
        dist_sq = dx**2 + dy**2  # 距离的平方
        dist_sq = np.sqrt(dist_sq + eps)  # 加上 eps 避免除零错误

        # 计算权重并存储在 bag_ori 中
        bag_ori[:, :, i] = 1 / (dist_sq**(2*alpha))

    # 归一化权重
    P_star1 = np.zeros((h, w), dtype=np.float32)
    P_star2 = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        P_star1 = P_star1 + bag_ori[:,:,i]*source_pts[i][0]
        P_star2 = P_star2 + bag_ori[:,:,i]*source_pts[i][1]

    sum_3D = np.sum(bag_ori,axis=2)
    P_star1 = P_star1/sum_3D
    P_star2 = P_star2/sum_3D

    Q_star1 = np.zeros((h, w), dtype=np.float32)
    Q_star2 = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        Q_star1 = Q_star1 + bag_ori[:,:,i]*target_pts[i][0]
        Q_star2 = Q_star2 + bag_ori[:,:,i]*target_pts[i][1]

    Q_star1 = Q_star1/sum_3D
    Q_star2 = Q_star2/sum_3D

    P_hat = np.zeros((n, h, w, 2), dtype=np.float32)  # 每个控制点对每个像素位置的影响
    Q_hat = np.zeros((n, h, w, 2), dtype=np.float32)

    for i in range(n):
        # 计算所有像素点相对于第 i 个控制点的偏移
        P_hat[i, :, :, 0] = source_pts[i][0] - P_star1  # x 方向
        P_hat[i, :, :, 1] = source_pts[i][1] - P_star2  # y 方向

    for i in range(n):
        P_hat[i, :, :, 0] = source_pts[i][0] - P_star1
        P_hat[i, :, :, 1] = source_pts[i][1] - P_star2
        Q_hat[i, :, :, 0] = target_pts[i][0] - Q_star1
        Q_hat[i, :, :, 1] = target_pts[i][1] - Q_star2


    # 初始化 Aj
    Aj = np.zeros((2, 2, h, w), dtype=np.float32)

    # 计算 Aj 矩阵
    for i in range(n):
        Aj[0, 0, :, :] += bag_ori[:, :, i] * P_hat[i, :, :, 0] * P_hat[i, :, :, 0]
        Aj[0, 1, :, :] += bag_ori[:, :, i] * P_hat[i, :, :, 0] * P_hat[i, :, :, 1]
        Aj[1, 0, :, :] += bag_ori[:, :, i] * P_hat[i, :, :, 1] * P_hat[i, :, :, 0]
        Aj[1, 1, :, :] += bag_ori[:, :, i] * P_hat[i, :, :, 1] * P_hat[i, :, :, 1]

    
    # 计算行列式 ad - bc
    a = Aj[0, 0, :, :]
    b = Aj[0, 1, :, :]
    c = Aj[1, 0, :, :]
    d = Aj[1, 1, :, :]
    
    # 计算行列式 ad - bc
    det = a * d - b * c
    
    # 检查行列式是否为零，避免除零
    det =det+ 1e-6  # 为了防止除零错误，设置非常小的值
    
    # 手动求逆
    Aj[0, 0, :, :] = d / det
    Aj[0, 1, :, :] = -b / det
    Aj[1, 0, :, :] = -c / det
    Aj[1, 1, :, :] = a / det

    x_indices = x_indices - P_star1
    y_indices = y_indices - P_star2

    A_j = np.zeros((h, w, n), dtype=np.float32)
    for j in range(n):
        A_j[:,:,j]=(x_indices*Aj[0,0,:,:]*bag_ori[:,:,j]*P_hat[j, :, :, 0]
        +x_indices*Aj[0,1,:,:]*bag_ori[:,:,j]*P_hat[j, :, :, 1]
        +y_indices*Aj[1,0,:,:]*bag_ori[:,:,j]*P_hat[j, :, :, 0]
        +y_indices*Aj[1,1,:,:]*bag_ori[:,:,j]*P_hat[j, :, :, 1]
        )

    result_array = np.zeros((2,h, w), dtype=np.float32)
    for j in range(n):
        result_array[0,:,:]=result_array[0,:,:]+A_j[:,:,j]*Q_hat[j, :, :, 0]
        result_array[1,:,:]=result_array[1,:,:]+A_j[:,:,j]*Q_hat[j, :, :, 1]
    
    result_array[0,:,:]=result_array[0,:,:]+Q_star1
    result_array[1,:,:]=result_array[1,:,:]+Q_star2

    # 渲染结果图像

    # 将 result_array 转换为映射的坐标
    map_x, map_y = result_array[0], result_array[1]

    # 使用 OpenCV 的 remap 函数重新映射图像像素
    warped_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    warped_image = np.array(warped_image)

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
