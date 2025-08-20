import numpy as np
import cv2


def func_edge_protect(img):
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    edge_h = 60
    edge_height = func_edge_height(img)
    max_val = np.max(edge_height)

    if max_val == 0:
        edge_threshold = 0
    else:
        edge_threshold = edge_h / max_val
        if edge_threshold > 0.8:
            edge_threshold = 0.8

    # Canny边缘检测
    edge_region = cv2.Canny(img.astype(np.uint8), edge_threshold * 50, edge_threshold * 100)

    # 膨胀操作
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_edge = cv2.dilate(edge_region, se)
    img_supedge = 1.0 - img_edge.astype(np.float64) / 255.0

    # 高斯滤波
    gaussian_kernal = cv2.getGaussianKernel(5, 0.8)
    gaussian_kernal = gaussian_kernal * gaussian_kernal.T
    edge_protect = cv2.filter2D(img_supedge, -1, gaussian_kernal)

    return edge_protect


def func_edge_height(img):
    # 梯度算子
    G1 = np.array([[0, 0, 0, 0, 0],
                   [1, 3, 8, 3, 1],
                   [0, 0, 0, 0, 0],
                   [-1, -3, -8, -3, -1],
                   [0, 0, 0, 0, 0]], dtype=np.float64)

    G2 = np.array([[0, 0, 1, 0, 0],
                   [0, 8, 3, 0, 0],
                   [1, 3, 0, -3, -1],
                   [0, 0, -3, -8, 0],
                   [0, 0, -1, 0, 0]], dtype=np.float64)

    G3 = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 3, 8, 0],
                   [-1, -3, 0, 3, 1],
                   [0, -8, -3, 0, 0],
                   [0, 0, -1, 0, 0]], dtype=np.float64)

    G4 = np.array([[0, 1, 0, -1, 0],
                   [0, 3, 0, -3, 0],
                   [0, 8, 0, -8, 0],
                   [0, 3, 0, -3, 0],
                   [0, 1, 0, -1, 0]], dtype=np.float64)

    # 计算梯度
    grad = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float64)
    grad[:, :, 0] = cv2.filter2D(img, -1, G1) / 16
    grad[:, :, 1] = cv2.filter2D(img, -1, G2) / 16
    grad[:, :, 2] = cv2.filter2D(img, -1, G3) / 16
    grad[:, :, 3] = cv2.filter2D(img, -1, G4) / 16

    # 计算最大梯度
    max_gard = np.max(np.abs(grad), axis=2)
    maxgard = max_gard[2:-2, 2:-2]  # 裁剪边界

    # 填充边界
    edge_height = cv2.copyMakeBorder(maxgard, 2, 2, 2, 2, cv2.BORDER_REFLECT)
    return edge_height