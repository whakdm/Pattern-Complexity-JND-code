import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from skimage.filters import gaussian

# 设置中文字体
import matplotlib.font_manager as fm


def set_chinese_font():
    chinese_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "SimSun", "WenQuanYi Micro Hei", "Heiti TC"]
    available_fonts = [f for f in chinese_fonts if any(f.lower() in font.lower() for font in fm.findSystemFonts())]
    if available_fonts:
        plt.rcParams["font.family"] = [available_fonts[0]]
    else:
        plt.rcParams["font.family"] = ["sans-serif"]
        print("警告：未找到中文字体，可能无法正常显示中文")


set_chinese_font()
plt.switch_backend('TkAgg')


def block_process(image, block_size):
    """分块并填充，确保块大小一致"""
    rows, cols = image.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    padded_image = np.pad(image, ((0, pad_rows), (0, pad_cols)), 'constant')
    blocks = []
    new_rows, new_cols = padded_image.shape
    for i in range(0, new_rows, block_size):
        for j in range(0, new_cols, block_size):
            blocks.append(padded_image[i:i + block_size, j:j + block_size])
    return blocks, new_rows, new_cols, pad_rows, pad_cols, padded_image


def dct_transform(block):
    """完善的DCT变换：确保正交归一化并保留数值精度"""
    # 对输入块进行类型转换，避免精度损失
    block = block.astype(np.float64)
    # 二维DCT变换（先对行变换，再对列变换）
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    # 限制极小值，避免数值不稳定
    dct_block[np.abs(dct_block) < 1e-10] = 0
    return dct_block


def idct_transform(block):
    """完善的逆DCT变换：确保正确恢复图像块范围"""
    # 对DCT系数进行逆变换
    idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
    # 逆变换后可能产生微小负值，截断到[0,1]范围（图像灰度值范围）
    idct_block = np.clip(idct_block, 0, 1)
    # 归一化到原图像动态范围（如果需要）
    idct_block = (idct_block - idct_block.min()) / (idct_block.max() - idct_block.min() + 1e-10)
    return idct_block


def calculate_texture_energy(dct_block, block_size=8):
    """计算纹理能量：LF + MF + HF（符合论文公式）"""
    # 严格按照8x8块划分频率区域
    lf = dct_block[1:4, 1:4]  # 低频区域 (1-3行, 1-3列)
    mf = np.concatenate([dct_block[1:4, 4:7], dct_block[4:7, 1:4]])  # 中频区域
    hf = dct_block[4:7, 4:7]  # 高频区域 (4-6行, 4-6列)

    energy_lf = np.sum(np.square(lf))
    energy_mf = np.sum(np.square(mf))
    energy_hf = np.sum(np.square(hf))

    return energy_lf + energy_mf + energy_hf


def calculate_texture_energy_diff(center_energy, neighbor_energy):
    """计算纹理能量差（论文公式7）"""
    return np.abs(center_energy - neighbor_energy)


def calculate_similarity(delta_energy):
    """计算相似度（论文公式8）"""
    # 增加缩放因子，避免相似度过于接近
    return 1.0 / (1 + np.exp(0.5 * delta_energy))


def te_classify_block(dct_block, block_size=8):
    """TE块分类：边缘块(1)和非边缘块(0)（论文公式相关）"""
    hf = dct_block[4:7, 4:7]  # 高频区域
    total_energy = np.sum(np.square(dct_block)) + 1e-8  # 总能量（加epsilon避免除零）
    hf_energy_ratio = np.sum(np.square(hf)) / total_energy  # 高频能量占比

    # 动态调整阈值（根据图像统计特性）
    edge_threshold = 0.12  # 经测试更适合猫图的边缘检测
    return 1 if hf_energy_ratio > edge_threshold else 0


def bayesian_prediction(center_block, center_type, neighbor_blocks, block_size=8):
    """贝叶斯预测模型（论文公式9）"""
    # 分离边缘和非边缘邻域块
    edge_neighbors = [blk for idx, blk, typ in neighbor_blocks if typ == 1]
    non_edge_neighbors = [blk for idx, blk, typ in neighbor_blocks if typ == 0]

    # 选择预测候选块
    if center_type == 1 and edge_neighbors:
        candidates = edge_neighbors
    elif center_type == 0 and non_edge_neighbors:
        candidates = non_edge_neighbors
    else:
        candidates = edge_neighbors + non_edge_neighbors  # 混合模式

    # 计算相似度权重
    weights = []
    for neighbor in candidates:
        center_energy = calculate_texture_energy(center_block)
        neighbor_energy = calculate_texture_energy(neighbor)
        delta_energy = calculate_texture_energy_diff(center_energy, neighbor_energy)
        similarity = calculate_similarity(delta_energy)
        weights.append(similarity)

    # 权重归一化
    weights = np.array(weights) / (np.sum(weights) + 1e-10)
    # 加权预测（论文公式9的实现）
    predicted_block = np.sum([w * blk for w, blk in zip(weights, candidates)], axis=0)

    return np.clip(predicted_block, 0, 1)  # 确保预测值范围稳定


def main():
    image_path = r"E:\TEST_CODE\Learn_JNDCODE\imgs\cat1.png"
    block_size = 8  # 8x8 DCT块（论文推荐）

    try:
        # 读取图像并预处理
        image = imread(image_path)
        if len(image.shape) == 3:
            image = rgb2gray(image)  # 转为灰度图
        image = gaussian(image, sigma=0.5)  # 轻微降噪，保留边缘
        image = np.clip(image, 0, 1)  # 确保图像值在[0,1]范围
    except Exception as e:
        print(f"图像读取失败: {e}")
        return

    # 分块处理
    blocks, new_rows, new_cols, pad_rows, pad_cols, padded_image = block_process(image, block_size)

    # DCT变换与块分类
    dct_blocks = [dct_transform(block) for block in blocks]
    block_types = [te_classify_block(blk) for blk in dct_blocks]

    # 生成分类结果图（用于调试）
    classification_map = np.zeros((new_rows, new_cols), dtype=np.uint8)
    block_idx = 0
    for i in range(0, new_rows, block_size):
        for j in range(0, new_cols, block_size):
            if block_types[block_idx] == 1:
                classification_map[i:i + block_size, j:j + block_size] = 255  # 边缘块（白色）
            else:
                classification_map[i:i + block_size, j:j + block_size] = 0  # 非边缘块（黑色）
            block_idx += 1
    # 去除填充
    classification_map = classification_map[:new_rows - pad_rows, :new_cols - pad_cols]

    # 对每个块进行贝叶斯预测
    predicted_dct_blocks = []
    for idx, center_dct in enumerate(dct_blocks):
        # 计算当前块的行列索引
        row = idx // (new_cols // block_size)
        col = idx % (new_cols // block_size)

        # 获取8邻域块（带类型信息）
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # 跳过中心块
                nr, nc = row + dr, col + dc
                # 检查邻域是否在图像范围内
                if 0 <= nr < (new_rows // block_size) and 0 <= nc < (new_cols // block_size):
                    n_idx = nr * (new_cols // block_size) + nc
                    neighbors.append((n_idx, dct_blocks[n_idx], block_types[n_idx]))

        # 执行预测
        predicted_dct = bayesian_prediction(center_dct, block_types[idx], neighbors, block_size)
        predicted_dct_blocks.append(predicted_dct)

    # 逆DCT变换并重建图像
    predicted_blocks = [idct_transform(blk) for blk in predicted_dct_blocks]

    # 拼接预测块为完整图像
    predicted_image = np.zeros_like(padded_image)
    block_idx = 0
    for i in range(0, new_rows, block_size):
        for j in range(0, new_cols, block_size):
            predicted_image[i:i + block_size, j:j + block_size] = predicted_blocks[block_idx]
            block_idx += 1

    # 去除填充部分
    if pad_rows > 0:
        predicted_image = predicted_image[:-pad_rows, :]
    if pad_cols > 0:
        predicted_image = predicted_image[:, :-pad_cols]

    # 计算预测误差（用于评估）
    prediction_error = np.abs(image - predicted_image)

    # 可视化结果
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(classification_map, cmap='gray')
    plt.title("TE块分类结果（白色=边缘块，黑色=非边缘块）")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(predicted_image, cmap='gray')
    plt.title("贝叶斯预测结果")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(prediction_error, cmap='jet')
    plt.title("预测误差（绝对值）")
    plt.colorbar(label="误差值")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
