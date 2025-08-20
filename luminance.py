import numpy as np
import cv2


def func_bg_lum_jnd(img0):
    min_lum = 32
    alpha = 0.7
    B = np.array([[1, 1, 1, 1, 1],
                  [1, 2, 2, 2, 1],
                  [1, 2, 0, 2, 1],
                  [1, 2, 2, 2, 1],
                  [1, 1, 1, 1, 1]], dtype=np.float64)

    # 滤波计算背景亮度
    bg_lum0 = np.floor(cv2.filter2D(img0, -1, B) / 32)
    bg_lum = func_bg_adjust(bg_lum0, min_lum)

    # 获取亮度JND阈值
    bg_jnd = lum_jnd()

    # 计算每个像素的亮度适应JND
    row, col = img0.shape
    jnd_lum = np.zeros((row, col))
    for x in range(row):
        for y in range(col):
            idx = int(bg_lum[x, y]) + 1
            if idx < 1:
                idx = 1
            elif idx > 256:
                idx = 256
            jnd_lum[x, y] = bg_jnd[idx - 1]

    return alpha * jnd_lum


def lum_jnd():
    bg_jnd = np.zeros(256)
    T0 = 17
    gamma = 3 / 128
    for k in range(256):
        lum = k
        if lum <= 127:
            bg_jnd[k] = T0 * (1 - np.sqrt(lum / 127)) + 3
        else:
            bg_jnd[k] = gamma * (lum - 127) + 3
    return bg_jnd


def func_bg_adjust(bg_lum0, min_lum):
    row, col = bg_lum0.shape
    bg_lum = bg_lum0.copy()
    for x in range(row):
        for y in range(col):
            if bg_lum[x, y] <= 127:
                bg_lum[x, y] = np.round(min_lum + bg_lum[x, y] * (127 - min_lum) / 127)
    return bg_lum