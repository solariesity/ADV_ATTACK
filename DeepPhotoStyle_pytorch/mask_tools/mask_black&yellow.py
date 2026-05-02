import numpy as np
from PIL import Image


def generate_masks(
    input_path: str,
    mask1_path: str,
    mask2_path: str,
    target_yellow: tuple = (250, 250, 30),  # 适配你这张图的亮黄色
    color_distance_threshold: float = 80.0,  # 宽松阈值，阴影偏暗的黄色也能识别
    alpha_threshold: int = 200,
) -> None:
    """生成黄色区域掩码和黑色区域掩码"""
    img = Image.open(input_path).convert("RGBA")
    img_arr = np.array(img)
    rgb, alpha = img_arr[:, :, :3], img_arr[:, :, 3]

    # 宽松的黄色判定：计算像素和目标黄色的颜色距离，距离小于阈值就算黄色
    r_dist = rgb[:, :, 0].astype(np.float32) - target_yellow[0]
    g_dist = rgb[:, :, 1].astype(np.float32) - target_yellow[1]
    b_dist = rgb[:, :, 2].astype(np.float32) - target_yellow[2]
    color_distance = np.sqrt(r_dist**2 + g_dist**2 + b_dist**2)
    yellow_mask = color_distance <= color_distance_threshold

    # 黑色掩码：非黄色 + 不透明（符合你要求）
    black_mask = (~yellow_mask) & (alpha >= alpha_threshold)

    # 保存掩码
    Image.fromarray((yellow_mask * 255).astype(np.uint8), mode="L").save(mask1_path)
    Image.fromarray((black_mask * 255).astype(np.uint8), mode="L").save(mask2_path)
    print(f"掩码生成完成！\n黄色区域：{mask1_path}\n黑色区域：{mask2_path}")


if __name__ == "__main__":
    # generate_masks(
    #     input_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/America.png",
    #     mask1_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/America_YellowMask.png",
    #     mask2_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/America_BlackMask.png",
    # )
    generate_masks(
        input_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/shixi.png",
        mask1_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/shixi_YellowMask.png",
        mask2_path="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/shixi_RedMask.png",
    )
