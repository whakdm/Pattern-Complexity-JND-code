import numpy as np
import cv2


def func_ori_cmlx_compute(img):
    cmlx_map = func_cmlx_num_compute(img)
    r = 3
    sig = 1
    fker = cv2.getGaussianKernel(r, sig)
    fker = fker * fker.T  # 2D高斯核
    cmlx_mat = cv2.filter2D(cmlx_map, -1, fker)
    return cmlx_mat


def func_cmlx_num_compute(img):
    r = 1
    nb = r * 8
    otr = 6

    # 梯度算子
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64) / 3
    ky = kx.T

    # 计算邻域采样点
    sps = np.zeros((nb, 2))
    as_ = 2 * np.pi / nb
    for i in range(nb):
        sps[i, 0] = -r * np.sin(i * as_)
        sps[i, 1] = r * np.cos(i * as_)

    # 扩展图像边界
    imgd = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
    row, col = imgd.shape

    # 计算梯度
    Gx = cv2.filter2D(imgd, -1, kx)
    Gy = cv2.filter2D(imgd, -1, ky)
    Cimg = np.sqrt(Gx ** 2 + Gy ** 2)

    # 有效像素标记
    Cvimg = np.zeros((row, col), dtype=np.float64)
    Cvimg[Cimg >= 5] = 1

    # 计算角度
    Oimg = np.round(np.arctan2(Gy, Gx) / np.pi * 180)
    Oimg[Oimg > 90] -= 180
    Oimg[Oimg < -90] += 180
    Oimg += 90  # 转换到[0, 180]
    Oimg[Cvimg == 0] = 180 + 2 * otr

    # 裁剪到原始尺寸
    Oimgc = Oimg[r:row - r, r:col - r]
    Cvimgc = Cvimg[r:row - r, r:col - r]

    # 标准化角度
    Oimg_norm = np.round(Oimg / (2 * otr))
    Oimgc_norm = np.round(Oimgc / (2 * otr))
    onum = int(np.round(180 / (2 * otr)) + 1)

    # 计算方向统计
    rows, cols = Oimgc_norm.shape
    ssr_val = np.zeros((rows, cols, onum + 1), dtype=np.float64)

    # 中心像素
    for x in range(onum + 1):
        ssr_val[:, :, x] = (Oimgc_norm == x).astype(np.float64)

    # 邻域像素
    for i in range(nb):
        dx = int(np.round(r + sps[i, 0]))
        dy = int(np.round(r + sps[i, 1]))
        Oimgn = Oimg_norm[dx:dx + rows, dy:dy + cols]
        for x in range(onum + 1):
            ssr_val[:, :, x] += (Oimgn == x).astype(np.float64)

    # 计算复杂度
    ssr_no_zero = (ssr_val != 0).astype(np.float64)
    cmlx = np.sum(ssr_no_zero, axis=2)
    cmlx[Cvimgc == 0] = 1

    # 边界处理
    cmlx[:r, :] = 1
    cmlx[-r:, :] = 1
    cmlx[:, :r] = 1
    cmlx[:, -r:] = 1

    return cmlx