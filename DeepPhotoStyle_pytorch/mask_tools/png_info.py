"""
PNG图像颜色提取工具
提取PNG图像中所有唯一的颜色
"""

from PIL import Image
import numpy as np


def get_png_colors(image_path):
    """
    读取PNG图像并返回所有唯一颜色

    参数:
        image_path: PNG图像文件路径

    返回:
        包含所有唯一颜色的集合，每个颜色为(R,G,B,A)元组
    """
    # 打开图像并确保为RGBA模式
    img = Image.open(image_path).convert("RGBA")

    # 转换为numpy数组并获取所有像素
    pixels = np.array(img).reshape(-1, 4)

    # 获取所有唯一颜色
    unique_colors = np.unique(pixels, axis=0)

    # 转换为颜色元组的集合
    return set(tuple(color) for color in unique_colors)


# 使用示例
if __name__ == "__main__":
    colors = get_png_colors("new_LP.png")
    print(f"找到 {len(colors)} 种唯一颜色")
    for color in list(colors)[:10]:  # 只显示前10种颜色
        print(f"  RGB({color[0]}, {color[1]}, {color[2]}) Alpha:{color[3]}")
