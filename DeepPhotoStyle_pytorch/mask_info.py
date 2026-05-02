from PIL import Image
import numpy as np


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


def analyze_mask(mask_path):
    """
    分析PNG格式的mask图片，返回其尺寸、像素取值等属性。

    参数:
        mask_path (str): mask图片的路径（如"mask.png"）

    返回:
        dict: 包含mask属性的字典（或错误信息）
    """
    try:
        # 1. 读取图片并检查格式
        with Image.open(mask_path) as img:
            if img.format != "PNG":
                raise ValueError("输入文件不是PNG格式，请检查文件类型！")

            # 2. 转换为NumPy数组（方便处理像素）
            img_array = np.array(img)

            # 3. 获取基本属性（尺寸、通道数）
            height, width = img_array.shape[:2]  # 注意：NumPy数组的shape是(高度, 宽度)
            channels = img_array.shape[2] if len(img_array.shape) == 3 else 1  # 通道数（1=灰度/二值，3=彩色，4=带alpha通道）

            # 4. 统计像素取值（处理alpha通道）
            if channels == 4:
                # 分离alpha通道（透明通道），忽略完全透明的像素（alpha=0）
                alpha_channel = img_array[:, :, 3]
                non_transparent_mask = alpha_channel > 0  # 非透明区域的掩码
                if np.any(non_transparent_mask):
                    # 提取非透明区域的RGB通道像素（忽略alpha）
                    rgb_pixels = img_array[non_transparent_mask, :3]
                    unique_values = np.unique(rgb_pixels)
                else:
                    unique_values = []  # 全透明图片
            else:
                # 灰度图/彩色图：直接统计所有像素
                unique_values = np.unique(img_array)

            # 5. 判断是否为二值图像（仅含两个不同像素值，如0和255）
            is_binary = len(unique_values) == 2

            # 6. 整理结果
            result = {
                "image_path": mask_path,
                "size": (width, height),  # 输出格式：(宽, 高)
                "channels": channels,  # 通道数（1=灰度，3=彩色，4=带alpha）
                "unique_pixel_values": unique_values.tolist(),  # 所有唯一像素值（如[0, 255]）
                "is_binary": is_binary,  # 是否为二值图像（True/False）
            }
            return result

    except Exception as e:
        # 捕获异常（如文件不存在、格式错误）
        return {"error": str(e)}


# ---------------------- 示例用法 ----------------------
if __name__ == "__main__":
    # 替换为你的mask图片路径（如"mask.png"）
    # mask_file = "/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/car/BMW_PaintMask25.png"
    mask_file = "/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/LP.png"
    # 分析mask属性
    mask_info = analyze_mask(mask_file)

    # 输出结果
    if "error" in mask_info:
        print(f"错误：{mask_info['error']}")
    else:
        print("mask属性分析结果：")
        print(f"图片路径：{mask_info['image_path']}")
        print(f"尺寸（宽×高）：{mask_info['size'][0]} × {mask_info['size'][1]}")
        print(f"通道数：{mask_info['channels']}")
        print(f"唯一像素值：{mask_info['unique_pixel_values']}")
        print(f"是否为二值图像：{mask_info['is_binary']}")
