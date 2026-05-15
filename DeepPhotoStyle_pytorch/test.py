import sys
import socket
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

import config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # ----------init------------
    ap = argparse.ArgumentParser()

    ap.add_argument("-s", "--style_image", required=True, help="name of the style image")

    ap.add_argument("-c", "--content_image", required=True, help="name of the content image")
    ap.add_argument("-v", "--vehicle", required=True, type=str, help="The name of the vehicle image")
    ap.add_argument(
        "-pm",
        "--paint-mask",
        required=True,
        type=str,
        help="""The types of the paint mask, e.g. '-1'/'01'/'02'/'03' 
        -1: 0.5 initilization and mask optimization
        -2: boarder mask optimization
         """,
    )
    ap.add_argument("--gpu", type=str, help="specify a GPU to use")

    ap.add_argument("--style-weight", "-sw", default=1000000, type=float, help="Style similarity weight")
    ap.add_argument("--content-weight", "-cw", default=1000, type=float, help="Content similarity weight")
    ap.add_argument("--tv-weight", "-tw", default=0.0001, type=float, help="Transform variant weight")
    ap.add_argument("--rl-weight", "-rw", default=10, type=float, help="Reality weight")
    ap.add_argument("--adv-weight", "-aw", default=1000000, type=float, help="Adversarial weight")
    ap.add_argument("--epsilon", "-eps", default=0.1, type=float, help="epsilon for l1-norm")
    ap.add_argument("--mask-weight", "-mw", default=1000, type=float, help="weight for paint mask")
    ap.add_argument("--l1-weight", "-l1w", default=300, type=float, help="l1 loss weight for perterbation")
    ap.add_argument("--steps", default=10000, type=int, help="total training steps")
    ap.add_argument("--learning-rate", "-lr", default=1, type=float, help="leanring rate")
    ap.add_argument("--batch-size", "-bs", default=6, type=int, help="optimization batch size")
    ap.add_argument("--l1-norm", dest="l1_norm", action="store_true", help="Wheather to use L1 Norm to find sensitive area")
    ap.add_argument("--random-scene", "-rs", action="store_true", help="Test whether we use different scene to train")
    ap.add_argument("--mask-step", "-ms", default=1, type=int, help="minimum mask unite size for mask optimization")
    ap.add_argument("--depth-model", "-dm", type=str, default="monodepth2", choices=["monodepth2", "depthhints", "manydepth"], help="select the depth model to be attacked")
    ap.add_argument("--adv-type", "-at", type=str, required=True, choices=["disp", "depth", "max_disp", "ratio_depth", "yolo"], help="select the adv loss type")
    ap.add_argument("--random-seed", "-seed", type=int, default=17, help="random seed in optimization")
    ap.add_argument("--log-postfix", "-lp", type=str, default="", help="log folder postfix")
    ap.add_argument("--late-start", action="store_true", help="start mask opimize from the second phase")
    ap.add_argument("--baseline", "-bl", type=str, default="proposed", choices=["baseline", "proposed"], help="Baseline or proposed method")
    ap.add_argument("--style-lambda", "-sl", default=1, type=float, help="Style transfer wrap weight")

    # added
    ap.add_argument("--end-learning-rate", "-elr", default=1 / 3, type=float, help="end leanring rate")
    ap.add_argument("--decay-steps", "-ds", default=2 / 3, type=float, help="decay steps")
    ap.add_argument("--device", "-d", default=0, type=float, help="0-cuda:0, 1-cuda:1")
    ap.add_argument("--decay-power", "-dp", default=0.9, type=float, help="power of learning rate")
    ap.add_argument("--nps-weight", "-nw", default=0.01, type=float, help="nps weight")
    ap.add_argument("--color-weight", "-clw", default=0.01, type=float, help="color weight")
    ap.add_argument("--color-weight-14", "-cw14", default=0.1 * 18, type=float, help="extra weight for color loss terms 1 and 4")
    ap.add_argument("--color-weight-2356", "-cw2356", default=1.0 * 18, type=float, help="weight for color loss terms 2, 3, 5 and 6")
    ap.add_argument("--original-weight", "-ow", default=0.01, type=float, help="original weight")
    ap.add_argument("--fixed-location", "-fl", default=1, type=float, help="fixed location")
    ap.add_argument("--class-lambda", "-cl", default=1, type=float, help="class lambda")
    ap.add_argument("--midu-weight", "-ml", default=10000, type=float, help="midu weight")
    ap.add_argument("--color-power", "-cp", default=1, type=int, choices=[1, 2], help="Color loss power: 1=mean (default), 2=squared")
    ap.add_argument("--save-gradcam", "-sgc", default=0, type=int, choices=[0, 1], help="Save Grad-CAM images: 1=save, 0=do not save")
    ap.add_argument("--official-yolo-tensor", "-oyt", default=0, type=int, choices=[0, 1], help="Use official-style tensor preprocess/NMS for YOLO tensor path: 1=enable, 0=disable")
    ap.add_argument(
        "--optimizer-type",
        "-ot",
        default="lbfgs",
        type=str,
        choices=["lbfgs", "adam"],
        help="optimizer used for input optimization",
    )

    args = vars(ap.parse_args())
    print(str(args))

    device_num = args["device"]
    if device_num == 0:
        config.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_num == 1:
        config.device0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ------custom module----

    import my_utils as utils
    from image_preprocess import prepare_dir, process_content_img, process_style_img, process_scene_img, process_car_img, image_to_car
    from image_preprocess import gen_content_path

    sys.path.append("seg")
    from model import *
    from my_yolov5.yolov5_model import import_yolov5s_model_1, tensor_to_image
    from PIL import Image

    command = "python " + " ".join(sys.argv)
    print("执行的命令是:", command)

    setup_seed(args["random_seed"])
    if args["random_scene"]:
        print("random_scene")
    else:
        print("fixed")

    style_image_name = args["style_image"]
    content_image_name = args["content_image"]
    if args["gpu"] != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    # -------------------------
    prepare_dir()
    style_img_resize, style_mask_np = process_style_img(style_image_name)
    content_img_resize, content_mask_np, content_white_mask_np, content_red_mask_np, content_black_mask_np, content_yellow_mask_np = process_content_img(content_image_name)

    # the following could be converted to data loader
    car_img_resize, car_mask_np, paint_mask_np = process_car_img(args["vehicle"], paintMask_no=args["paint_mask"], mask_step=args["mask_step"])
    scene_img_crop = process_scene_img("VW01.png")
    test_scene_img = process_scene_img("VW01.png")

    # scene_img_crop = process_scene_img('000001.png')
    # test_scene_img = process_scene_img('000001.png')

    print("Computing Laplacian matrix of content image")
    content_image_path = os.path.join(gen_content_path, content_image_name)
    L = utils.compute_lap(content_image_path)
    print()

    width_s, height_s = style_img_resize.size
    width_c, height_c = content_img_resize.size

    style_mask_tensor = torch.from_numpy(style_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    car_mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_mask_tensor = torch.from_numpy(content_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_white_mask_tensor = torch.from_numpy(content_white_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_red_mask_tensor = torch.from_numpy(content_red_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_black_mask_tensor = torch.from_numpy(content_black_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_yellow_mask_tensor = torch.from_numpy(content_yellow_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    # paint_mask_tensor = torch.from_numpy(paint_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    # paint_mask_np_inf = np.arctanh((paint_mask_np - 0.5) * (2 - 1e-7))
    # paint_mask_inf = torch.from_numpy(paint_mask_np_inf).unsqueeze(0).float().to(config.device0).requires_grad_(True)
    # paint_mask_boarders = torch.tensor([0, car_mask_tensor.size()[2], 0, car_mask_tensor.size()[1]]).float().to(config.device0).requires_grad_(True)

    paint_mask_init = utils.get_mask_source(args["paint_mask"], car_mask_tensor.size(), paint_mask_np, args)
    paint_mask_tensor = utils.get_mask_target(args["paint_mask"], car_mask_tensor.size(), paint_mask_init)

    # 1*3*320*1024
    style_img = utils.image_to_tensor(style_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    content_img = utils.image_to_tensor(content_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    car_img = utils.image_to_tensor(car_img_resize)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    scene_img = utils.image_to_tensor(scene_img_crop)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    test_scene_img = utils.image_to_tensor(test_scene_img)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)

    # Logger
    # log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname())
    # log_dir = os.path.join('/data/cheng443/depth_atk', 'logs', datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname() + '_CH')
    log_dir = os.path.join(utils.log_dir, "logs", datetime.datetime.now().strftime("%b%d_%H-%M-%S_") + args["log_postfix"])
    os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)
    logger.add_text("args/CLI_params", str(args), 0)

    logger.add_image("input/imgs/style_image", style_img[0], 0)
    logger.add_image("input/imgs/car_img", car_img[0], 0)
    logger.add_image("input/imgs/content_img", content_img[0], 0)
    logger.add_image("input/masks/style_mask", style_mask_tensor, 0)
    logger.add_image("input/masks/car_mask", car_mask_tensor, 0)
    logger.add_image("input/masks/paint_mask", paint_mask_tensor, 0)
    logger.add_image("input/masks/content_mask", content_mask_tensor, 0)
    logger.add_image("input/masks/content_mask", content_white_mask_tensor, 0)
    logger.add_image("input/masks/content_mask", content_red_mask_tensor, 0)
    logger.add_image("input/masks/content_mask", content_black_mask_tensor, 0)
    logger.add_image("input/masks/content_mask", content_yellow_mask_tensor, 0)

    # 将命令保存到文件中
    command_file = os.path.join(log_dir, "command.txt")
    with open(command_file, "w") as f:
        f.write(command)

    # -------------------------
    # Eval() means change the model in eval mode and requires_grad = False means
    # the parameters of cnn are frozen.
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(config.device0).eval()
    for param in cnn.parameters():
        param.requires_grad = False

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(config.device0)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(config.device0)

    # Two different initialization ways
    # if args['l1_norm']:
    if True:
        input_img = content_img.clone()
        input_img.requires_grad = True  # 确保可以计算梯度
    else:
        input_img = torch.randn(1, 3, height_c, width_c).to(config.device0)
    #
    print(content_white_mask_np.size == 0)
    print(content_red_mask_np.size == 0)
    print(content_black_mask_np.size == 0)
    print(content_yellow_mask_np.size == 0)
    # if np.all(content_white_mask_np == 0):
    #     print("掩码全为0")
    # else:
    #     print("掩码不全为0")
    # if np.all(content_red_mask_np == 0):
    #     print("掩码全为0")
    # else:
    #     print("掩码不全为0")
    # if np.all(content_black_mask_np == 0):
    #     print("掩码全为0")
    # else:
    #     print("掩码不全为0")
    # if np.all(content_yellow_mask_np == 0):
    #     print("掩码全为0")
    # else:
    #     print("掩码不全为0")
    # exit()

    output, yolo_model_2, yolo_result, adv_scene_ret = run_style_transfer(
        logger,
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        car_img,
        scene_img,
        test_scene_img,
        style_mask_tensor,
        content_mask_tensor,
        content_white_mask_tensor,
        content_red_mask_tensor,
        content_black_mask_tensor,
        content_yellow_mask_tensor,
        paint_mask_init,
        car_mask_tensor,
        log_dir,
        L,
        args,
    )
    yolo_result.save(os.path.join(log_dir, "yolo_result.jpg"))
    print("Style transfer completed")

    # utils.save_pic(output, "deep_style_tranfer")
    print("output")
    print(type(output))
    print(output.shape)
    print(type(content_mask_np))
    print(type(content_white_mask_np))
    print(type(content_red_mask_np))
    print(type(content_black_mask_np))
    print(type(content_yellow_mask_np))

    # adv_car_output = output * paint_mask_tensor.unsqueeze(0) + car_img * (1-paint_mask_tensor.unsqueeze(0))

    logger.add_image("Output/whole_texture_transfer", output[0], 0)
    print()

    paint_mask_tensor = utils.get_mask_target(args["paint_mask"], car_mask_tensor.size(), paint_mask_init)

    # utils.save_pic(adv_car_output, f'adv_car_output')

    if args["paint_mask"] == "-2":
        output = utils.texture_to_car_size(output, car_img.size())
        adv_car_output = output * paint_mask_tensor.unsqueeze(0) + car_img * (1 - paint_mask_tensor.unsqueeze(0))
    if args["paint_mask"] != "-2":
        adv_car_output = image_to_car(output, paint_mask_tensor, content_mask_tensor, car_img)
        # pass
    adv_scene_out, car_scene_out, scene_car_mask, scene_paint_mask = attach_car_to_scene(
        test_scene_img, adv_car_output, car_img, car_mask_tensor, args["batch_size"], paint_mask_tensor, args["vehicle"]
    )

    utils.save_pic(adv_scene_out[[0]], f"adv_scene_output", log_dir=log_dir)
    utils.save_pic(car_scene_out[[0]], f"car_scene_output", log_dir=log_dir)
    utils.save_pic(adv_car_output[[0]], f"adv_car_output", log_dir=log_dir)

    logger.add_image("Output/Adv_scene", adv_scene_out[0], 0)
    logger.add_image("Output/Car_scene", car_scene_out[0], 0)
    logger.add_image("Output/Adv_car", adv_car_output[0], 0)

    yolo_model = import_yolov5s_model_1(device_num=device_num)
    adv_scene_result = yolo_model(tensor_to_image(adv_scene_ret))

    adv_scene_disp = yolo_model(tensor_to_image(adv_scene_out))

    cv2.imwrite(os.path.join(log_dir, "eval_adv_scene_disp.jpg"), cv2.cvtColor(adv_scene_disp.render()[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(log_dir, "adv_scene_result.jpg"), cv2.cvtColor(adv_scene_result.render()[0], cv2.COLOR_RGB2BGR))

    print("Done!")

    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    # 生成网格张量（一行6列，自动归一化）

    print("adv_scene_out.shape:", adv_scene_out.shape)
    adv_scene_tensor = make_grid(adv_scene_out, nrow=6, normalize=True)  # 形状 [3, 320, 6*1024]
    car_scene_tensor = make_grid(car_scene_out, nrow=6, normalize=True)  # 形状 [3, 320, 6*1024]

    # 转换为 PIL 图像
    adv_scene_img = to_pil_image(adv_scene_tensor)
    car_scene_img = to_pil_image(car_scene_tensor)
