import numpy as np
import cv2
from luminance import func_bg_lum_jnd
from contrast import func_luminance_contrast
from complexity import func_ori_cmlx_compute
from edge import func_edge_protect
from random_num import func_randnum


def func_JND_modeling_pattern_complexity(img):
    # 转换为double类型
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    # 亮度适应
    jnd_LA = func_bg_lum_jnd(img)

    # 亮度对比度掩蔽
    L_c = func_luminance_contrast(img)
    a1 = 0.115 * 16
    a2 = 26
    jnd_LC = (a1 * L_c ** 2.4) / (L_c ** 2 + a2 ** 2)

    # 内容复杂度
    P_c = func_ori_cmlx_compute(img)
    a3 = 0.3
    a4 = 2.7
    a5 = 1
    C_t = (a3 * P_c ** a4) / (P_c ** 2 + a5 ** 2)
    jnd_PM = L_c * C_t

    # 边缘保护
    edge_protect = func_edge_protect(img)
    jnd_PM_p = jnd_PM * edge_protect

    # 视觉掩蔽
    jnd_VM = np.maximum(jnd_LC, jnd_PM_p)

    # JND映射
    jnd_map = jnd_LA + jnd_VM - 0.3 * np.minimum(jnd_LA, jnd_VM)

    # 注入噪声
    row, col = img.shape
    randmat = func_randnum(row, col)
    adjuster = 0.9
    img_jnd = (img + adjuster * randmat * jnd_map).astype(np.uint8)

    # 计算MSE
    mse_val = np.mean((img_jnd.astype(np.float64) - img) ** 2)
    print(f"MSE = {mse_val:.3f}")

    return img_jnd, jnd_map, jnd_LA, jnd_VM, P_c