import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from jnd_models import func_JND_with_complexity, func_JND_without_complexity


def calculate_jnd_vs_original_ssim(original_img, jnd_map):
    """计算JND图与原始图像之间的SSIM值"""
    # 确保原始图像为灰度图且为uint8类型
    original = original_img.astype(np.uint8)

    # 归一化JND图到0-255范围（与原始图像保持一致的数据范围）
    jnd_norm = cv2.normalize(jnd_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 计算SSIM
    ssim_value = ssim(
        original,
        jnd_norm,
        data_range=255,  # 原始图像和归一化后的JND图范围都是0-255
        multichannel=False
    )
    return ssim_value, jnd_norm


# 创建保存结果的文件夹
save_dir = "jnd_vs_original_ssim_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 加载原始图像
img = cv2.imread('imgs/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("图像文件未找到，请检查路径是否正确")

# 保存原始图像
cv2.imwrite(os.path.join(save_dir, "original_image.png"), img)

# 生成两种模型的JND图
# 带模式复杂度的模型
_, jnd_map_with, _, _, _ = func_JND_with_complexity(img.copy())
# 不带模式复杂度的模型
_, jnd_map_without, _, _ = func_JND_without_complexity(img.copy())

# 计算带模式复杂度的JND图与原始图像的SSIM
ssim_with, jnd_with_norm = calculate_jnd_vs_original_ssim(img, jnd_map_with)
# 计算不带模式复杂度的JND图与原始图像的SSIM
ssim_without, jnd_without_norm = calculate_jnd_vs_original_ssim(img, jnd_map_without)

# 保存JND图用于对比
cv2.imwrite(os.path.join(save_dir, "jnd_with_complexity.png"), jnd_with_norm)
cv2.imwrite(os.path.join(save_dir, "jnd_without_complexity.png"), jnd_without_norm)

# 输出结果
print(f"带模式复杂度的JND图与原始图像的SSIM值: {ssim_with:.6f}")
print(f"不带模式复杂度的JND图与原始图像的SSIM值: {ssim_without:.6f}")

# 结果分析
print("\n结果分析:")
print("1. JND图与原始图像的SSIM值反映了两者在结构上的关联性")
print("2. 通常情况下，这两个SSIM值都会较低（远小于1），因为:")
print("   - 原始图像表示的是亮度信息")
print("   - JND图表示的是噪声容忍阈值，与亮度信息不完全相关")
print("3. 带模式复杂度的JND图可能与原始图像有更高的结构关联性，因为它更紧密地依赖图像内容复杂度")
