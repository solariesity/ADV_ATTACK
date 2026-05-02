import argparse
import os
import random

import numpy as np
import torch

import config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("-s", "--style_image", required=True, help="name of the style image")
    ap.add_argument("-v", "--vehicle", default="BMW.png", type=str, help="The name of the vehicle image")
    ap.add_argument(
        "-pm",
        "--paint-mask",
        required=True,
        type=str,
        help="The paint mask id, e.g. 11/12/13 or negative values for optimized masks.",
    )
    ap.add_argument("--gpu", type=str, help="specify a GPU to use")
    ap.add_argument("--mask-step", "-ms", default=1, type=int, help="minimum mask unit size for mask optimization")
    ap.add_argument("--device", "-d", default=0, type=float, help="0-cuda:0, 1-cuda:1")
    ap.add_argument("--random-seed", "-seed", type=int, default=17, help="random seed")
    ap.add_argument("--scene-image", default="VW01.png", type=str, help="scene image name")

    args = vars(ap.parse_args())
    args["content_image"] = args["style_image"]
    print(str(args))

    if args["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    device_num = int(args["device"])
    if device_num == 0:
        config.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_num == 1:
        config.device0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        config.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    setup_seed(args["random_seed"])

    import my_utils as utils
    from attach_cars import attach_car_to_scene_fixed
    from image_preprocess import image_to_car
    from image_preprocess import prepare_dir, process_car_img, process_content_img, process_scene_img, process_style_img

    output_dir = os.path.join(os.getcwd(), "output_test", "test_car")
    os.makedirs(output_dir, exist_ok=True)

    prepare_dir()

    # Keep the same preprocessing flow as test.py.
    process_style_img(args["style_image"])
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

    car_mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_mask_tensor = torch.from_numpy(content_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)

    # These tensors are loaded to mirror test.py's full preprocessing path.
    torch.from_numpy(content_white_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    torch.from_numpy(content_red_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    torch.from_numpy(content_black_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    torch.from_numpy(content_yellow_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)

    paint_mask_init = utils.get_mask_source(args["paint_mask"], car_mask_tensor.size(), paint_mask_np, args)
    paint_mask_tensor = utils.get_mask_target(args["paint_mask"], car_mask_tensor.size(), paint_mask_init)

    content_img = utils.image_to_tensor(content_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    car_img = utils.image_to_tensor(car_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    scene_img = utils.image_to_tensor(scene_img_crop)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)

    # test.py initializes input_img from content_img before optimization.
    input_img = content_img.clone()
    adv_car_img = image_to_car(input_img, paint_mask_tensor, content_mask_tensor, car_img)
    adv_scene_img, _, _ = attach_car_to_scene_fixed(
        scene_img,
        adv_car_img,
        car_img,
        car_mask_tensor,
        object_name=args["vehicle"],
    )

    prefix = "{}__{}__{}__{}".format(
        os.path.splitext(args["content_image"])[0],
        os.path.splitext(args["vehicle"])[0],
        args["paint_mask"],
        os.path.splitext(args["scene_image"])[0],
    )

    utils.save_pic(adv_car_img, f"{prefix}_patch_on_car", output_dir)
    utils.save_pic(adv_scene_img, f"{prefix}_patch_car_on_scene", output_dir)

    print(f"Saved outputs to: {output_dir}")
