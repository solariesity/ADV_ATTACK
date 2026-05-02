# %%
import os
import PIL.Image as pil
from PIL import ImageOps
import numpy as np
import my_utils as utils
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import config

src_content_path = os.path.join(os.getcwd(), "asset", "src_img", "content")
src_style_path = os.path.join(os.getcwd(), "asset", "src_img", "style")
src_scene_path = os.path.join(os.getcwd(), "asset", "src_img", "scene")
src_car_path = os.path.join(os.getcwd(), "asset", "src_img", "car")

gen_content_path = os.path.join(os.getcwd(), "asset", "gen_img", "content")
gen_style_path = os.path.join(os.getcwd(), "asset", "gen_img", "style")
gen_scene_path = os.path.join(os.getcwd(), "asset", "gen_img", "scene")
gen_car_path = os.path.join(os.getcwd(), "asset", "gen_img", "car")

car_img_width = 600
scene_size = (1024, 320)  # width, height


# %%
def prepare_dir():
    if not os.path.exists(gen_content_path):
        os.makedirs(gen_content_path)
    if not os.path.exists(gen_style_path):
        os.makedirs(gen_style_path)
    if not os.path.exists(gen_scene_path):
        os.makedirs(gen_scene_path)
    if not os.path.exists(gen_car_path):
        os.makedirs(gen_car_path)


def process_img(img_name, output_w, image_type: str):
    if image_type == "style":
        img_path = os.path.join(src_style_path, img_name)
        img_out_path = os.path.join(gen_style_path, img_name)
    elif image_type == "content":
        img_path = os.path.join(src_content_path, img_name)
        img_out_path = os.path.join(gen_content_path, img_name)
    elif image_type == "car":
        img_path = os.path.join(src_car_path, img_name)
        img_out_path = os.path.join(gen_car_path, img_name)
    if not os.path.exists(img_path):
        raise RuntimeError("image '%s' doesn't exist" % img_path)
    style_img = pil.open(img_path)
    original_w, original_h = style_img.size
    print("Image original size (w, h): (%d, %d)" % (original_w, original_h))

    output_h = int(output_w / original_w * original_h)
    style_img_resize = style_img.resize((output_w, output_h))
    style_img_resize.save(img_out_path)
    print("Output image size", style_img_resize.size)
    return style_img_resize, output_w, output_h


def process_mask(mask_name, output_w, output_h, image_type: str):
    if image_type == "style":
        mask_path = os.path.join(src_style_path, mask_name)
        mask_out_path = os.path.join(gen_style_path, mask_name)
    elif image_type == "content":
        mask_path = os.path.join(src_content_path, mask_name)
        mask_out_path = os.path.join(gen_content_path, mask_name)
    elif image_type == "car":
        mask_path = os.path.join(src_car_path, mask_name)
        mask_out_path = os.path.join(gen_car_path, mask_name)
    if not os.path.exists(mask_path):
        img_mask_np = np.ones((output_h, output_w), dtype=int)
        print(f"The mask [{mask_name}] doesn't exist, using the whole image...")
        # img_mask_np = np.zeros((output_h, output_w), dtype=int)
        # print(f"The mask [{mask_name}] doesn't exist, using full zero mask...")
    else:
        img_mask = ImageOps.grayscale(pil.open(mask_path))
        img_mask_np = np.array(img_mask.resize((output_w, output_h))) / 255.0
        img_mask_np[img_mask_np > 0.5] = 1
        img_mask_np[img_mask_np <= 0.5] = 0
        img_mask_np = img_mask_np.astype(int)
    pil.fromarray((img_mask_np * 255).astype(np.uint8), "L").save(mask_out_path)
    return img_mask_np


def process_style_img(img_name):
    ext_split = os.path.splitext(img_name)
    style_img_resize, w, h = process_img(img_name, car_img_width, "style")
    img_mask_np = process_mask(ext_split[0] + "_StyleMask" + ext_split[1], w, h, "style")
    assert style_img_resize.size[::-1] == img_mask_np.shape
    return style_img_resize, img_mask_np


def process_content_img(img_name):
    ext_split = os.path.splitext(img_name)
    content_img_resize, w, h = process_img(img_name, car_img_width, "content")
    content_mask_np = process_mask(ext_split[0] + "_ContentMask" + ext_split[1], w, h, "content")
    content_white_mask_np = process_mask(ext_split[0] + "_WhiteMask" + ext_split[1], w, h, "content")
    content_red_mask_np = process_mask(ext_split[0] + "_RedMask" + ext_split[1], w, h, "content")
    content_black_mask_np = process_mask(ext_split[0] + "_BlackMask" + ext_split[1], w, h, "content")
    content_yellow_mask_np = process_mask(ext_split[0] + "_YellowMask" + ext_split[1], w, h, "content")
    assert content_img_resize.size[::-1] == content_mask_np.shape
    return content_img_resize, content_mask_np, content_white_mask_np, content_red_mask_np, content_black_mask_np, content_yellow_mask_np


def process_car_img(img_name, paintMask_no: str, mask_step: int = 1):
    ext_split = os.path.splitext(img_name)
    car_img_resize, w, h = process_img(img_name, car_img_width, "car")
    car_mask_np = process_mask(ext_split[0] + "_CarMask" + ext_split[1], w, h, "car")
    # if paintMask_no == '-1' or paintMask_no == '-2' : # half mask
    if int(paintMask_no) < 0:  # half mask
        mask_shape = [(i // mask_step) for i in car_mask_np.shape]
        # paint_mask_np = np.random.random(mask_shape)
        paint_mask_np = np.ones(mask_shape) * 0.5
        paint_mask_np = np.clip(paint_mask_np, 0.0, 1.0)
    else:
        paint_mask_np = process_mask(ext_split[0] + "_PaintMask" + paintMask_no + ext_split[1], w, h, "car")
    print(ext_split[0] + "_PaintMask" + paintMask_no + ext_split[1])
    assert car_img_resize.size[::-1] == car_mask_np.shape
    return car_img_resize, car_mask_np, paint_mask_np


def process_scene_img(img_name):
    scene_img = pil.open(os.path.join(src_scene_path, img_name))
    original_w, original_h = scene_img.size
    new_w, new_h = scene_size
    left = (original_w - new_w) // 2
    right = left + new_w
    top = original_h - new_h
    bottom = original_h
    scene_img_crop = scene_img.crop((left, top, right, bottom))
    assert scene_size == scene_img_crop.size
    scene_img_crop.save(os.path.join(gen_scene_path, img_name))
    return scene_img_crop


def image_to_car(output, paint_mask_tensor, content_mask_tensor, car_img):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = config.device0
    # 找到白色矩形的边界
    mask = paint_mask_tensor.squeeze(0).to(device)  # 移动到设备
    rows, cols = torch.where(mask == 1)  # 找到白色像素的坐标
    if rows.numel() == 0 or cols.numel() == 0:
        raise ValueError("No white pixels found in paint_mask_tensor")

    top = rows.min().item()
    bottom = rows.max().item()
    left = cols.min().item()
    right = cols.max().item()

    # print(top, bottom, left, right)

    # 计算白色矩形的尺寸
    new_height = bottom - top + 1
    new_width = right - left + 1

    current_aspect_ratio = new_width / new_height

    # 获取 output 的尺寸
    if output.dim() == 4:  # [1, 3, H, W]
        _, _, output_height, output_width = output.shape
    else:  # [3, H, W]
        _, output_height, output_width = output.shape

    # target_aspect_ratio = 5 / 3
    # target_aspect_ratio = 194 / 211
    target_aspect_ratio = output_width / output_height

    # 根据目标比例调整尺寸（保持面积近似）
    if current_aspect_ratio < target_aspect_ratio:
        # 当前更宽，以宽度为基准调整高度
        new_height = int(new_width * 1 / target_aspect_ratio)
        bottom = new_height - 1 + top
    else:
        # 当前更高，以高度为基准调整宽度
        new_width = int(new_height * target_aspect_ratio)
        right = new_width - 1 + left

    # 定义缩放变换
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((new_height, new_width)), transforms.ToTensor()])  # 转换为PIL图像  # 缩放到白色矩形的尺寸  # 转换回张量

    # 应用缩放变换
    # output_resized = transform(output.squeeze(0).cpu()).unsqueeze(0).to(device)  # 注意：这里先将output移动到CPU进行PIL处理（因为PIL不支持GPU tensor），然后再移动回设备
    # 可微的 resize 操作
    import torch.nn.functional as F

    # 可微的 resize 操作（直接使用 PyTorch 函数）
    def differentiable_resize(tensor, size):
        """
        tensor: 输入张量 [C, H, W] 或 [B, C, H, W]
        size: 目标尺寸 (height, width)
        """
        return (
            F.interpolate(tensor.unsqueeze(0) if tensor.dim() == 3 else tensor, size=size, mode="bilinear", align_corners=False).squeeze(0)  # 统一为 [B, C, H, W]  # 双线性插值（可微）  # 避免边界伪影
            if tensor.dim() == 3
            else tensor
        )  # 恢复原始维度

    # 直接对 output 进行可微缩放（无需 .cpu() 或 .to(device)）
    output_resized = differentiable_resize(output.squeeze(0), (new_height, new_width)).unsqueeze(0)

    # print(output.requires_grad)
    # sys.exit()

    # 创建一个全黑的背景张量
    full_size_output = torch.zeros((1, 3, 520, 600), dtype=output_resized.dtype, device=device)  # 指定设备

    # 将缩放后的output放到白色矩形区域
    full_size_output[:, :, top : bottom + 1, left : right + 1] = output_resized

    # print(full_size_output.requires_grad)
    # print(output.requires_grad)
    # sys.exit()
    # print("Full size output shape:", full_size_output.shape)

    # utils.save_pic(full_size_output, 'full_size_output')

    # 将content_mask_np转换为PIL图像（因为0-1值，乘以255转为0-255范围方便处理）
    pil_content_mask = Image.fromarray((content_mask_tensor.cpu().detach().squeeze(0).numpy() * 255).astype(np.uint8))

    # pil_content_mask = Image.fromarray((content_mask_np * 255).astype(np.uint8))

    # 定义缩放变换（这里不进行ToTensor，因为后面还要处理通道）
    transform_resize = transforms.Resize((new_height, new_width))

    # 应用缩放变换
    resized_pil_content_mask = transform_resize(pil_content_mask)

    # 将缩放后的PIL图像转换为numpy数组并扩展维度以匹配通道数（假设3通道）
    resized_content_mask_np = np.array(resized_pil_content_mask).astype(np.float32) / 255.0  # 转回0-1范围
    resized_content_mask_np = np.tile(resized_content_mask_np[np.newaxis, :, :], (3, 1, 1))  # 扩展通道维度

    # 创建一个全黑的背景张量[1, 3, 520, 600]
    full_size_content_mask = torch.zeros((1, 3, 520, 600), dtype=torch.float32, device=device)  # 指定设备

    # 将缩放并处理好通道的content_mask放到白色矩形区域
    full_size_content_mask[:, :, top : bottom + 1, left : right + 1] = torch.from_numpy(resized_content_mask_np).to(device)  # 移动到设备

    # print("Final content_mask shape:", full_size_content_mask.shape)

    # utils.save_pic(full_size_content_mask, "full_size_content_mask")

    car_img = car_img.to(device)  # 确保car_img也在设备上

    adv_car_output = full_size_output * full_size_content_mask + car_img * (1 - full_size_content_mask)  # 不需要unsqueeze(0)，因为full_size_content_mask已经是[1, 3, 520, 600]

    # print(adv_car_output.requires_grad)
    # print(output.requires_grad)
    # sys.exit()
    # utils.save_pic(adv_car_output, 'adv_car_output')
    return adv_car_output


# %%
if __name__ == "__main__":
    prepare_dir()
    process_style_img("Dirty_Back.png")
    process_content_img("Warnning.png")
    process_car_img("BMW.png", paintMask_no="12")
    # process_car_img("Wall.png", paintMask_no='01')
    process_scene_img("0000000090.png")
    # process_scene_img("0000000090.png")
    # process_scene_img("000043.png")
# %%
