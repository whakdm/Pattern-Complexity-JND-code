import cv2
import numpy as np
import matplotlib.pyplot as plt
from jnd_model import func_JND_modeling_pattern_complexity
import os

save_dir="jnd_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 加载图像
img = cv2.imread('imgs/lena.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("图像文件未找到，请检查路径")

# 计算JND相关结果
img_jnd, jnd_map, jnd_LA, jnd_VM, complexity_map = func_JND_modeling_pattern_complexity(img)


# cv2.imwrite(os.path.join(save_dir, "jnd_result.png"), img)
plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.savefig(os.path.join(save_dir, "original_image_plt.png"), bbox_inches='tight')  #


# cv2.imwrite(os.path.join(save_dir, "jnd_result.png"), jnd_LA)
plt.figure()
plt.imshow(jnd_LA, cmap='gray')
plt.title('Luminance Adaption')
plt.savefig(os.path.join(save_dir, "Luminance Adaption.png"), bbox_inches='tight')  #


plt.figure()
plt.imshow(complexity_map, cmap='gray')
plt.title('Complexity Map')
plt.savefig(os.path.join(save_dir, "Complexity Map.png"), bbox_inches='tight')  #


plt.figure()
plt.imshow(jnd_VM, cmap='gray')
plt.title('Visual Masking')
plt.savefig(os.path.join(save_dir, "Visual Masking.png"), bbox_inches='tight')  #


plt.figure()
plt.imshow(img_jnd, cmap='gray')
plt.title('JND Noise Contaminated Image')
plt.savefig(os.path.join(save_dir, "JND Noise Contaminated Image.png"), bbox_inches='tight')  #

plt.show()


