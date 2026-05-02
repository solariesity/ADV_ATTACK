import numpy as np
from PIL import Image


def orange_to_yellow(input_path: str, output_path: str, target_yellow=(255, 255, 0), orange_threshold=(200, 120, 0)):  # 标准黄色  # 判断橙色的阈值下界
    """
    将图像中的橙色区域替换为黄色

    参数：
    - input_path: 输入图片路径
    - output_path: 输出图片路径
    - target_yellow: 替换成的黄色 (RGB)
    - orange_threshold: 橙色判断阈值 (R, G, B)
    """

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    # 拆分通道
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    # 橙色区域判断（经验规则：R高，G中，B低）
    mask = (r > orange_threshold[0]) & (g > orange_threshold[1]) & (b < orange_threshold[2] + 80)

    # 替换为黄色
    img_np[mask] = target_yellow

    # 保存
    result = Image.fromarray(img_np)
    result.save(output_path)

    print("处理完成，已保存到:", output_path)


if __name__ == "__main__":
    orange_to_yellow(
        input_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/Australia.png",
        output_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/Australia_2.png",
    )  # 替换为你的图片路径
