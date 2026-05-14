import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

import config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_pil(image, output_dir, filename):
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path


def save_mask_np(mask_np, output_dir, filename):
    mask_img = Image.fromarray((mask_np.astype(np.float32) * 255).astype(np.uint8))
    return save_pil(mask_img, output_dir, filename)


def save_tensor_image(tensor, output_dir, filename):
    path = os.path.join(output_dir, filename)
    image_tensor = tensor.detach().cpu().clone()
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.clamp(0, 1)
    to_pil_image(image_tensor).save(path)
    return path


def save_grid_image(batch_tensor, output_dir, filename, nrow=6):
    grid_tensor = make_grid(batch_tensor.detach().cpu(), nrow=nrow, normalize=True)
    path = os.path.join(output_dir, filename)
    to_pil_image(grid_tensor).save(path)
    return path


def write_summary(summary_lines, output_dir):
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    return summary_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--style_image", required=True, help="name of the style image")
    ap.add_argument("-c", "--content_image", required=True, help="name of the content image")
    ap.add_argument("-v", "--vehicle", required=True, type=str, help="The name of the vehicle image")
    ap.add_argument(
        "-pm",
        "--paint-mask",
        required=True,
        type=str,
        help="The paint mask id, e.g. 11/12/13 or negative values for optimized masks.",
    )
    ap.add_argument("--gpu", type=str, help="specify a GPU to use")
    ap.add_argument("--mask-step", "-ms", default=1, type=int, help="minimum mask unit size for mask optimization")
    ap.add_argument("--device", "-d", default=0, type=int, help="0-cuda:0, 1-cuda:1")
    ap.add_argument("--batch-size", "-bs", default=6, type=int, help="batch size used in attach_car_to_scene")
    ap.add_argument("--random-seed", "-seed", type=int, default=17, help="random seed")
    ap.add_argument("--scene-image", default="VW01.png", type=str, help="scene image name")
    ap.add_argument("--fixed-location", "-fl", default=1, type=int, choices=[0, 1], help="1=fixed training placement, 0=robust training placement")

    args = vars(ap.parse_args())
    print(str(args))

    if args["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    if args["device"] == 0:
        config.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args["device"] == 1:
        config.device0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        config.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    setup_seed(args["random_seed"])

    import my_utils as utils
    from attach_cars import attach_car_to_scene, attach_car_to_scene_fixed, attach_car_to_scene_Robustness_training
    from image_preprocess import image_to_car
    from image_preprocess import prepare_dir, process_car_img, process_content_img, process_scene_img, process_style_img

    output_dir = os.path.join(os.getcwd(), "test_output", "test")
    os.makedirs(output_dir, exist_ok=True)

    prepare_dir()

    style_img_resize, style_mask_np = process_style_img(args["style_image"])
    (
        content_img_resize,
        content_mask_np,
        content_white_mask_np,
        content_red_mask_np,
        content_black_mask_np,
        content_yellow_mask_np,
    ) = process_content_img(args["content_image"])
    car_img_resize, car_mask_np, paint_mask_np = process_car_img(
        args["vehicle"],
        paintMask_no=args["paint_mask"],
        mask_step=args["mask_step"],
    )
    scene_img_crop = process_scene_img(args["scene_image"])
    test_scene_img_crop = process_scene_img(args["scene_image"])

    summary_lines = []
    summary_lines.append(f"device: {config.device0}")
    summary_lines.append(f"style processed size (W,H): {style_img_resize.size}")
    summary_lines.append(f"content processed size (W,H): {content_img_resize.size}")
    summary_lines.append(f"car processed size (W,H): {car_img_resize.size}")
    summary_lines.append(f"scene processed size (W,H): {scene_img_crop.size}")

    save_pil(style_img_resize, output_dir, "01_style_processed.png")
    save_mask_np(style_mask_np, output_dir, "02_style_mask.png")
    save_pil(content_img_resize, output_dir, "03_content_processed.png")
    save_mask_np(content_mask_np, output_dir, "04_content_mask.png")
    save_mask_np(content_white_mask_np, output_dir, "05_content_white_mask.png")
    save_mask_np(content_red_mask_np, output_dir, "06_content_red_mask.png")
    save_mask_np(content_black_mask_np, output_dir, "07_content_black_mask.png")
    save_mask_np(content_yellow_mask_np, output_dir, "08_content_yellow_mask.png")
    save_pil(car_img_resize, output_dir, "09_car_processed.png")
    save_mask_np(car_mask_np, output_dir, "10_car_mask.png")
    save_mask_np(paint_mask_np, output_dir, "11_paint_mask_raw.png")
    save_pil(scene_img_crop, output_dir, "12_scene_processed.png")
    save_pil(test_scene_img_crop, output_dir, "13_test_scene_processed.png")

    style_mask_tensor = torch.from_numpy(style_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    car_mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_mask_tensor = torch.from_numpy(content_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_white_mask_tensor = torch.from_numpy(content_white_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_red_mask_tensor = torch.from_numpy(content_red_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_black_mask_tensor = torch.from_numpy(content_black_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_yellow_mask_tensor = torch.from_numpy(content_yellow_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)

    paint_mask_init = utils.get_mask_source(args["paint_mask"], car_mask_tensor.size(), paint_mask_np, args)
    paint_mask_tensor = utils.get_mask_target(args["paint_mask"], car_mask_tensor.size(), paint_mask_init)

    style_img = utils.image_to_tensor(style_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    content_img = utils.image_to_tensor(content_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    car_img = utils.image_to_tensor(car_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    scene_img = utils.image_to_tensor(scene_img_crop)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    test_scene_img = utils.image_to_tensor(test_scene_img_crop)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)

    summary_lines.append(f"style tensor shape: {tuple(style_img.shape)}")
    summary_lines.append(f"content tensor shape: {tuple(content_img.shape)}")
    summary_lines.append(f"car tensor shape: {tuple(car_img.shape)}")
    summary_lines.append(f"scene tensor shape: {tuple(scene_img.shape)}")
    summary_lines.append(f"test scene tensor shape: {tuple(test_scene_img.shape)}")
    summary_lines.append(f"style mask tensor shape: {tuple(style_mask_tensor.shape)}")
    summary_lines.append(f"content mask tensor shape: {tuple(content_mask_tensor.shape)}")
    summary_lines.append(f"car mask tensor shape: {tuple(car_mask_tensor.shape)}")
    summary_lines.append(f"paint mask tensor shape: {tuple(paint_mask_tensor.shape)}")

    save_tensor_image(style_img, output_dir, "14_style_tensor.png")
    save_tensor_image(content_img, output_dir, "15_content_tensor.png")
    save_tensor_image(car_img, output_dir, "16_car_tensor.png")
    save_tensor_image(scene_img, output_dir, "17_scene_tensor.png")
    save_tensor_image(test_scene_img, output_dir, "18_test_scene_tensor.png")
    save_tensor_image(style_mask_tensor, output_dir, "19_style_mask_tensor.png")
    save_tensor_image(content_mask_tensor, output_dir, "20_content_mask_tensor.png")
    save_tensor_image(content_white_mask_tensor, output_dir, "21_content_white_mask_tensor.png")
    save_tensor_image(content_red_mask_tensor, output_dir, "22_content_red_mask_tensor.png")
    save_tensor_image(content_black_mask_tensor, output_dir, "23_content_black_mask_tensor.png")
    save_tensor_image(content_yellow_mask_tensor, output_dir, "24_content_yellow_mask_tensor.png")
    save_tensor_image(car_mask_tensor, output_dir, "25_car_mask_tensor.png")
    save_tensor_image(paint_mask_tensor, output_dir, "26_paint_mask_tensor.png")

    input_img = content_img.clone()
    save_tensor_image(input_img, output_dir, "27_input_img_init.png")

    if args["paint_mask"] == "-2":
        input_img_resize = utils.texture_to_car_size(input_img, car_img.size())
        save_tensor_image(input_img_resize, output_dir, "28_input_img_resized_for_car.png")
        adv_car_image = input_img_resize * paint_mask_tensor.unsqueeze(0) + car_img * (1 - paint_mask_tensor.unsqueeze(0))
    else:
        adv_car_image = image_to_car(input_img, paint_mask_tensor, content_mask_tensor, car_img)

    summary_lines.append(f"adv_car_image shape: {tuple(adv_car_image.shape)}")
    save_tensor_image(adv_car_image, output_dir, "29_adv_car_image.png")

    if bool(args["fixed_location"]):
        adv_scene_train, car_scene_train, scene_obj_mask_train = attach_car_to_scene_fixed(
            scene_img,
            adv_car_image,
            car_img,
            car_mask_tensor,
            object_name=args["vehicle"],
        )
        summary_lines.append(f"adv_scene_train shape: {tuple(adv_scene_train.shape)}")
        summary_lines.append(f"car_scene_train shape: {tuple(car_scene_train.shape)}")
        summary_lines.append(f"scene_obj_mask_train shape: {tuple(scene_obj_mask_train.shape)}")
        save_tensor_image(adv_scene_train[[0]], output_dir, "30_train_adv_scene_fixed.png")
        save_tensor_image(car_scene_train[[0]], output_dir, "31_train_car_scene_fixed.png")
        save_tensor_image(scene_obj_mask_train[[0]], output_dir, "32_train_scene_obj_mask_fixed.png")
    else:
        adv_scene_train, car_scene_train, scene_obj_mask_train, scene_paint_mask_train = attach_car_to_scene_Robustness_training(
            scene_img,
            adv_car_image,
            car_img,
            car_mask_tensor,
            args["batch_size"],
            paint_mask_tensor,
            args["vehicle"],
        )
        summary_lines.append(f"adv_scene_train shape: {tuple(adv_scene_train.shape)}")
        summary_lines.append(f"car_scene_train shape: {tuple(car_scene_train.shape)}")
        summary_lines.append(f"scene_obj_mask_train shape: {tuple(scene_obj_mask_train.shape)}")
        summary_lines.append(f"scene_paint_mask_train shape: {tuple(scene_paint_mask_train.shape)}")
        save_tensor_image(adv_scene_train[[0]], output_dir, "30_train_adv_scene_robust.png")
        save_tensor_image(car_scene_train[[0]], output_dir, "31_train_car_scene_robust.png")
        save_tensor_image(scene_obj_mask_train[[0]], output_dir, "32_train_scene_obj_mask_robust.png")
        save_tensor_image(scene_paint_mask_train[[0]], output_dir, "33_train_scene_paint_mask_robust.png")

    if args["paint_mask"] == "-2":
        adv_car_output = utils.texture_to_car_size(input_img, car_img.size()) * paint_mask_tensor.unsqueeze(0) + car_img * (1 - paint_mask_tensor.unsqueeze(0))
    else:
        adv_car_output = image_to_car(input_img, paint_mask_tensor, content_mask_tensor, car_img)

    adv_scene_out, car_scene_out, scene_car_mask, scene_paint_mask = attach_car_to_scene(
        test_scene_img,
        adv_car_output,
        car_img,
        car_mask_tensor,
        args["batch_size"],
        paint_mask_tensor,
        args["vehicle"],
    )

    summary_lines.append(f"adv_car_output shape: {tuple(adv_car_output.shape)}")
    summary_lines.append(f"adv_scene_out shape: {tuple(adv_scene_out.shape)}")
    summary_lines.append(f"car_scene_out shape: {tuple(car_scene_out.shape)}")
    summary_lines.append(f"scene_car_mask shape: {tuple(scene_car_mask.shape)}")
    summary_lines.append(f"scene_paint_mask shape: {tuple(scene_paint_mask.shape)}")

    save_tensor_image(adv_car_output, output_dir, "34_adv_car_output.png")
    save_tensor_image(adv_scene_out[[0]], output_dir, "35_eval_adv_scene_first.png")
    save_tensor_image(car_scene_out[[0]], output_dir, "36_eval_car_scene_first.png")
    save_tensor_image(scene_car_mask[[0]], output_dir, "37_eval_scene_car_mask_first.png")
    save_tensor_image(scene_paint_mask[[0]], output_dir, "38_eval_scene_paint_mask_first.png")
    save_grid_image(adv_scene_out, output_dir, "39_eval_adv_scene_grid.png", nrow=args["batch_size"])
    save_grid_image(car_scene_out, output_dir, "40_eval_car_scene_grid.png", nrow=args["batch_size"])
    save_grid_image(scene_car_mask, output_dir, "41_eval_scene_car_mask_grid.png", nrow=args["batch_size"])
    save_grid_image(scene_paint_mask, output_dir, "42_eval_scene_paint_mask_grid.png", nrow=args["batch_size"])

    summary_path = write_summary(summary_lines, output_dir)

    print(f"Saved process-only outputs to: {output_dir}")
    print(f"Saved summary to: {summary_path}")
