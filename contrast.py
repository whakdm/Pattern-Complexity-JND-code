import numpy as np
import cv2


def func_luminance_contrast(img):
    R = 2
    ker = np.ones((2 * R + 1, 2 * R + 1), dtype=np.float64) / ((2 * R + 1) ** 2)

    # 计算均值
    mean_mask = cv2.filter2D(img, -1, ker)
    mean_img_sqr = mean_mask ** 2

    # 计算平方的均值
    img_sqr = img ** 2
    mean_sqr_img = cv2.filter2D(img_sqr, -1, ker)

    # 计算方差
    var_mask = mean_sqr_img - mean_img_sqr
    var_mask[var_mask < 0] = 0

    # 边界处理
    row, col = img.shape
    valid_mask = np.zeros((row, col), dtype=np.float64)
    valid_mask[R + 1:row - R, R + 1:col - R] = 1
    var_mask = var_mask * valid_mask

    return np.sqrt(var_mask)