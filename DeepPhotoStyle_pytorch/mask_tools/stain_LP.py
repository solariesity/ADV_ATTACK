from PIL import Image, ImageEnhance


def add_stain(base_path, stain_path, output_path, stain_opacity=0.8, base_saturation=0.4, base_brightness=0.8):
    """
    在基础图片上叠加污渍图层

    Args:
        base_path (str): 基础图片路径
        stain_path (str): 污渍图片路径
        output_path (str): 输出图片路径
        stain_opacity (float): 污渍透明度 [0-1]
        base_saturation (float): 基础图片饱和度调整 [0-1]
        base_brightness (float): 基础图片亮度调整 [0-1]
    """
    base = Image.open(base_path).convert("RGBA")
    base = ImageEnhance.Color(base).enhance(base_saturation)
    base = ImageEnhance.Brightness(base).enhance(base_brightness)

    stain = Image.open(stain_path).convert("RGBA").resize(base.size)
    if stain_opacity < 1.0:
        stain.putdata([(r, g, b, int(a * stain_opacity)) for r, g, b, a in stain.getdata()])

    Image.alpha_composite(base, stain).convert("RGB").save(output_path)


def apply_content_mask(dirty_path, mask_path, output_path):
    """
    将内容掩码应用到污渍图片

    Args:
        dirty_path (str): 污渍图片路径
        mask_path (str): 掩码图片路径（白色表示保留内容，黑色表示透明）
        output_path (str): 输出图片路径
    """
    dirty = Image.open(dirty_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(dirty.size)

    dirty.putalpha(mask)
    dirty.save(output_path)


if __name__ == "__main__":
    # 生成污渍图片
    add_stain(
        base_path="new_LP.png",  # 基础图片：原始的L形红色图像
        stain_path="stain.png",  # 污渍图片：透明背景的污渍纹理
        output_path="dirty_new_LP.png",  # 输出图片：添加污渍后的结果
        stain_opacity=0.6,  # 污渍透明度：60%不透明度（更自然）
        base_saturation=0.75,  # 基础图片饱和度：降低25%（轻微变灰）
        base_brightness=0.9,  # 基础图片亮度：降低10%（轻微变暗）
    )

    # 应用内容掩码
    apply_content_mask(
        dirty_path="dirty_new_LP.png",  # 污渍图片：上一步生成的带污渍图像
        mask_path="new_LP_ContentMask.png",  # 掩码图片：内容区域掩码（白色保留，黑色透明）
        output_path="dirty_new_LP.png",  # 输出图片：应用掩码后的结果
    )
