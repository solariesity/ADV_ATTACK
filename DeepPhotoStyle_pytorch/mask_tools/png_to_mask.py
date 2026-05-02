import cv2
import numpy as np


def png_to_mask(input_path: str, output_path: str) -> None:
    """
    从PNG图像的Alpha通道生成二值化掩码

    Args:
        input_path: 输入PNG文件路径（需包含Alpha通道）
        output_path: 输出掩码文件保存路径
    """
    # 读取PNG图像（包含Alpha通道）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"无法读取图像文件: {input_path}")

    if img.ndim < 3 or img.shape[2] < 4:
        raise ValueError("输入图像必须是带有Alpha通道的PNG格式")

    # 提取Alpha通道并二值化
    _, mask = cv2.threshold(img[:, :, 3], 1, 255, cv2.THRESH_BINARY)

    # 保存掩码图像
    if not cv2.imwrite(output_path, mask):
        raise ValueError(f"无法保存掩码图像到: {output_path}")

    print(f"掩码已保存至: {output_path}")


def image_to_mask(input_path: str, output_path: str) -> None:
    """
    根据输入图像类型生成二值化掩码
    - 如果是带Alpha通道的PNG：提取Alpha通道
    - 如果是JPG或不带Alpha的图片：生成同大小的全白掩码

    Args:
        input_path: 输入图像文件路径
        output_path: 输出掩码文件保存路径
    """
    # 读取图像（包含Alpha通道）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"无法读取图像文件: {input_path}")

    height, width = img.shape[:2]

    # 判断是否有Alpha通道 (PNG且通道数为4)
    has_alpha = img.ndim == 3 and img.shape[2] == 4

    if has_alpha:
        print("检测到Alpha通道，正在提取掩码...")
        # 提取Alpha通道并二值化 (阈值设为1，即只要不是全透明就是前景)
        _, mask = cv2.threshold(img[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    else:
        print(f"未检测到Alpha通道 (或为JPG)，正在生成 {width}x{height} 的全白掩码...")
        # 生成全白掩码 (数据类型需设为uint8，否则保存时会出错)
        mask = np.ones((height, width), dtype=np.uint8) * 255

    # 保存掩码图像
    if not cv2.imwrite(output_path, mask):
        raise ValueError(f"无法保存掩码图像到: {output_path}")

    print(f"掩码已保存至: {output_path}")


if __name__ == "__main__":
    try:
        image_to_mask(
            input_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/shixi.png",
            output_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/shixi_ContentMask.png",
        )
    except Exception as e:
        print(f"处理失败: {e}")
