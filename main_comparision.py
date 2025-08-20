import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from jnd_models import func_JND_with_complexity, func_JND_without_complexity




# 创建保存结果的文件夹
save_dir = "jnd_comparison_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建保存文件夹: {save_dir}")

# 加载图像
img = cv2.imread(r'D:\MY_PRO\test\imgs\cat1.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("图像文件未找到，请检查路径是否正确")

# 计算两种JND模型的结果
# 1. 带模式复杂度的完整模型
img_jnd_complex, jnd_map_complex, jnd_LA_complex, jnd_VM_complex, complexity_map = func_JND_with_complexity(img)

# 2. 不带模式复杂度的简化模型
img_jnd_simple, jnd_map_simple, jnd_LA_simple, jnd_VM_simple = func_JND_without_complexity(img)

# 显示并保存对比结果
plt.figure(figsize=(15, 10))

# 原始图像
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "original_image.png"), img)

# 带模式复杂度的结果
plt.subplot(3, 3, 2)
plt.imshow(img_jnd_complex, cmap='gray')
plt.title('With Complexity: Noisy Image')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "with_complexity_noisy.png"), img_jnd_complex)

# 不带模式复杂度的结果
plt.subplot(3, 3, 3)
plt.imshow(img_jnd_simple, cmap='gray')
plt.title('Without Complexity: Noisy Image')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "without_complexity_noisy.png"), img_jnd_simple)

# JND映射对比
jnd_map_complex_norm = cv2.normalize(jnd_map_complex, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
jnd_map_simple_norm = cv2.normalize(jnd_map_simple, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(3, 3, 5)
plt.imshow(jnd_map_complex_norm, cmap='gray')
plt.title('With Complexity: JND Map')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "with_complexity_jnd_map.png"), jnd_map_complex_norm)

plt.subplot(3, 3, 6)
plt.imshow(jnd_map_simple_norm, cmap='gray')
plt.title('Without Complexity: JND Map')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "without_complexity_jnd_map.png"), jnd_map_simple_norm)

# 视觉掩蔽对比
jnd_VM_complex_norm = cv2.normalize(jnd_VM_complex, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
jnd_VM_simple_norm = cv2.normalize(jnd_VM_simple, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(3, 3, 8)
plt.imshow(jnd_VM_complex_norm, cmap='gray')
plt.title('With Complexity: Visual Masking')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "with_complexity_masking.png"), jnd_VM_complex_norm)

plt.subplot(3, 3, 9)
plt.imshow(jnd_VM_simple_norm, cmap='gray')
plt.title('Without Complexity: Visual Masking')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "without_complexity_masking.png"), jnd_VM_simple_norm)

# 模式复杂度图（仅完整模型有）
complexity_norm = cv2.normalize(complexity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.subplot(3, 3, 4)
plt.imshow(complexity_norm, cmap='gray')
plt.title('Pattern Complexity Map')
plt.axis('off')
cv2.imwrite(os.path.join(save_dir, "complexity_map.png"), complexity_norm)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "comparison_summary.png"), bbox_inches='tight')
plt.show()

print(f"所有对比结果已保存到 {os.path.abspath(save_dir)} 文件夹")
