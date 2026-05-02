from PIL import Image
import numpy as np


def create_mask(output_path, width=1563, height=1355, channels=1, foreground_value=255, background_value=0, shape="rectangle", box=None):
    """
    创建二值mask图像并保存为PNG

    参数:
        output_path (str): 输出文件路径
        width (int): 图像宽度（默认1563）
        height (int): 图像高度（默认1355）
        channels (int): 通道数（默认1）
        foreground_value (int): 前景像素值（默认255）
        background_value (int): 背景像素值（默认0）
        shape (str): 前景形状（'rectangle'矩形/'ellipse'椭圆，默认矩形）
        box (tuple): 自定义前景区域坐标 (left, top, right, bottom)。如果为None，则使用中心矩形。
    """
    # 创建背景（全为background_value）
    if channels == 1:
        img_array = np.full((height, width), background_value, dtype=np.uint8)
    else:
        img_array = np.full((height, width, channels), background_value, dtype=np.uint8)

    # 创建前景区域
    if box:
        # 使用用户指定的坐标
        left, top, right, bottom = box
        fg_width = right - left
        fg_height = bottom - top
    else:
        # 默认创建中心位置前景区域（宽高的80%）
        center_x, center_y = width // 2, height // 2
        fg_width, fg_height = int(width * 0.8), int(height * 0.8)
        left = center_x - fg_width // 2
        top = center_y - fg_height // 2
        right = left + fg_width
        bottom = top + fg_height

    # 计算椭圆中心点（矩形中心）
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    # 创建前景
    if shape == "rectangle":
        if channels == 1:
            img_array[top:bottom, left:right] = foreground_value
        else:
            # 对于彩色图像，每个通道设置相同的值
            img_array[top:bottom, left:right] = [foreground_value] * channels

    elif shape == "ellipse":
        # 创建椭圆掩码
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x) / (fg_width / 2)) ** 2 + ((y - center_y) / (fg_height / 2)) ** 2 <= 1

        if channels == 1:
            img_array[mask] = foreground_value
        else:
            # 为每个通道设置前景值
            for c in range(channels):
                img_array[:, :, c] = np.where(mask, foreground_value, img_array[:, :, c])

    # 创建并保存图像
    img = Image.fromarray(img_array)
    img.save(output_path, "PNG")
    print(f"Mask已成功保存至: {output_path}")
    print(f"尺寸: {width}×{height} | 通道数: {channels}")
    print(f"前景位置: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"前景值: {foreground_value} | 背景值: {background_value}")


# 示例使用
if __name__ == "__main__":
    # 使用默认值创建mask（与你的示例相同）左上右下
    create_mask("/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/car/BMW_PaintMask37.png", box=(630, 660, 1330, 980))
    # 自定义参数创建mask
    # create_mask(output_path="mask.png", width=1563, height=1355, channels=1, foreground_value=255, background_value=0, shape="rectangle")
