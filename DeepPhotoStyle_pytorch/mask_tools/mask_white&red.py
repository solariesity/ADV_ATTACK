import numpy as np
from PIL import Image


def generate_masks(
    input_path: str,
    mask1_path: str,
    mask2_path: str,
    white_rgb: tuple = (230, 230, 230),
    alpha_threshold: int = 200,
) -> None:
    """生成白色区域掩码和红色区域掩码"""
    img = Image.open(input_path).convert("RGBA")
    img_arr = np.array(img)
    rgb, alpha = img_arr[:, :, :3], img_arr[:, :, 3]

    # 生成白色掩码
    white_mask = (rgb >= white_rgb).all(axis=2)

    # 生成红色掩码（非白色且不透明）
    red_mask = ~white_mask & (alpha >= alpha_threshold)

    # 保存掩码
    Image.fromarray((white_mask * 255).astype(np.uint8), mode="L").save(mask1_path)
    Image.fromarray((red_mask * 255).astype(np.uint8), mode="L").save(mask2_path)
    print(f"掩码生成完成！\n白色区域：{mask1_path}\n红色区域：{mask2_path}")


if __name__ == "__main__":
    generate_masks(
        input_path="new_LP.png",
        mask1_path="new_LP_3.png",
        mask2_path="new_LP_4.png",
        white_rgb=(230, 230, 230),
        alpha_threshold=200,
    )
