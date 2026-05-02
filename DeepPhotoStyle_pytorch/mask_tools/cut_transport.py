from PIL import Image


def crop_transparent(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")

    # 获取 alpha 通道
    alpha = img.split()[-1]

    # 找到非透明区域的 bounding box
    bbox = alpha.getbbox()

    if bbox:
        cropped = img.crop(bbox)
        cropped.save(output_path)
    else:
        print("图片是全透明的")


crop_transparent("/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/asset/src_img/content/shixi.png", "/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/asset/src_img/content/shixi_2.png")
