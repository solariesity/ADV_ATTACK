from PIL import Image
import os
import numpy as np


def jpg_to_png(jpg_path):
    """
    将JPG图像转换为PNG格式并保存到相同目录

    参数:
    jpg_path (str): 输入JPG文件的路径

    返回:
    str: 输出的PNG文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(jpg_path):
            raise FileNotFoundError(f"文件不存在: {jpg_path}")

        # 检查文件扩展名
        if not jpg_path.lower().endswith((".jpg", ".jpeg")):
            raise ValueError("输入文件必须是JPG格式")

        # 打开JPG图像
        with Image.open(jpg_path) as img:
            # 转换为RGB模式（处理可能存在的透明度问题）
            if img.mode in ("RGBA", "LA"):
                # 如果有透明度，使用白色背景
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # 构建输出路径
            base_path = os.path.splitext(jpg_path)[0]
            png_path = base_path + ".png"

            # 保存为PNG
            img.save(png_path, "PNG")

            print(f"转换成功: {png_path}")
            return png_path

    except Exception as e:
        print(f"转换失败: {str(e)}")
        return None


def create_white_mask(output_path, width=194, height=211):
    """
    生成一个全白的mask PNG图像

    参数:
    output_path (str): 输出PNG文件的路径
    width (int): 图像宽度，默认为194
    height (int): 图像高度，默认为211

    返回:
    str: 输出文件的路径
    """
    try:
        # 创建全白图像数组 (RGB模式，白色为(255, 255, 255))
        white_array = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 创建图像
        white_image = Image.fromarray(white_array, "RGB")

        # 保存为PNG
        white_image.save(output_path, "PNG")

        print(f"全白mask已生成: {output_path}")
        return output_path

    except Exception as e:
        print(f"生成失败: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # result = jpg_to_png("/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/LP.jpg")
    # if result:
    #     print(f"输出文件: {result}")

    create_white_mask("/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/asset/src_img/content/White.png")
