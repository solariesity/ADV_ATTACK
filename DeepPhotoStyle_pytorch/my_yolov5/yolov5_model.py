import os
import sys
import time
from http.client import RemoteDisconnected

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import cv2
import torch
import torchvision.transforms as transforms

import config

try:
    from .yolov5_draw import *
    from .yolov5_show import *
    from .yolov5_diff import *
except ImportError:
    from yolov5_draw import *
    from yolov5_show import *
    from yolov5_diff import *


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU
device = config.device0


def import_yolov5s_model_1(device_num=0, scene_size=None, model_type="yolov5s"):
    """
    导入YOLOv5模型（带简单重试机制）
    检测图片用
    只尝试从本地加载模型，失败后自动重试最多10次
    """
    saved_model_path = f"../models/yolov5s_1_{device_num}.pth"

    if os.path.exists(saved_model_path):
        yolov5_source_path = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        if yolov5_source_path not in sys.path:
            sys.path.append(yolov5_source_path)
        yolov5_model = torch.load(saved_model_path)
        print(f"模型已从 {saved_model_path} 加载。")
        return yolov5_model
    else:
        max_retries = 10
        retry_delay = 5  # 每次重试间隔5秒

        for attempt in range(max_retries):
            try:
                # 尝试加载本地模型
                yolov5_model = torch.hub.load("ultralytics/yolov5", "custom", path="../models/yolov5s.pt", autoshape=True)

                # 配置模型参数
                yolov5_model.conf = 0.1
                yolov5_model.iou = 0.45
                yolov5_model.agnostic = True  # 合并所有类别的重叠框
                yolov5_model.eval()
                # saved_model_path = "../models/yolov5s_1.pth"
                torch.save(yolov5_model, saved_model_path)
                return yolov5_model

            except (RemoteDisconnected, Exception) as e:
                if attempt < max_retries - 1:
                    print(f"模型加载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"无法加载YOLOv5模型，已达到最大重试次数 {max_retries}。错误: {str(e)}")


def import_yolov5s_model_2(device_num=0, scene_size=None, model_type="yolov5s"):
    """
    导入YOLOv5模型（带简单重试机制）
    检测张量用
    只尝试从本地加载模型，失败后自动重试最多10次
    """
    saved_model_path = f"../models/yolov5s_2_{device_num}.pth"

    if os.path.exists(saved_model_path):
        yolov5_source_path = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        if yolov5_source_path not in sys.path:
            sys.path.append(yolov5_source_path)
        yolov5_model = torch.load(saved_model_path)
        print(f"模型已从 {saved_model_path} 加载。")
        return yolov5_model
    else:
        max_retries = 10
        retry_delay = 5  # 每次重试间隔5秒

        for attempt in range(max_retries):
            try:
                # 尝试加载本地模型
                yolov5_model = torch.hub.load("ultralytics/yolov5", "custom", path="../models/yolov5s.pt", autoshape=False).to(device)  # 使用path参数指定模型路径

                # 配置模型参数
                yolov5_model.conf = 0.1
                yolov5_model.eval()
                # saved_model_path = "../models/yolov5s_2.pth"
                torch.save(yolov5_model, saved_model_path)
                return yolov5_model

            except (RemoteDisconnected, Exception) as e:
                if attempt < max_retries - 1:
                    print(f"模型加载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"无法加载YOLOv5模型，已达到最大重试次数 {max_retries}。错误: {str(e)}")


def tensor_to_image(tensor):
    """将4D/3D张量转换为PIL图像，自动处理batch维度"""
    if tensor.ndim == 4:  # [B,C,H,W] -> 取第一个图像
        tensor = tensor[0]  # 或使用 tensor.squeeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:  # [1,H,W] -> [H,W]
        tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)


# 使用示例
if __name__ == "__main__":
    # 加载模型
    model = import_yolov5s_model_1()

    # img_path = "./test_output/zidane.jpg"
    # img_path = "D:/hyj/lab/mde_attack/test/Jun12_16-35-39_mono_car_Rob_disp/adv_scene_output.png"
    # img_path = "test_output/bus.jpg"
    # img_path = "optimized_image_1.jpg"
    # img_path = "/data/hyj/advpatch/MDE_Attack/log/logs/Jul12_00-00-12_mono_car_Rob_disp/adv_scene_output.png" #0.24
    # img_path = "/data/hyj/advpatch/MDE_Attack/log/logs/Jul11_17-56-23_mono_car_Rob_disp/adv_scene_output.png"
    # img_path = "/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/test_output/image.png"
    img_path = "/home/hyj/data/log12/logs/Dec08_05-52-42_mono_car_Rob_disp/yolo_result.jpg"

    results = model(img_path)

    df = results.pandas().xyxy[0]
    print("\nDataFrame 格式结果:\n", df)

    for index, row in df.iterrows():
        print(f"检测到: {row['name']} (ID={row['class']}), " f"置信度: {row['confidence']:.2f}, " f"位置: [{row['xmin']:.0f}, {row['ymin']:.0f}, {row['xmax']:.0f}, {row['ymax']:.0f}]")

    # sys.exit()

    # results.render()[0]

    # 获取结果图像（RGB格式）
    output_img = results.render()[0]  # 获取第一个图像的渲染结果
    # 转换为BGR格式（OpenCV默认格式）
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    # 保存图像
    cv2.imwrite("./test_output/1.jpg", output_img_bgr)

    sys.exit()

    img_path1 = "./test_output/bus.jpg"
    results1 = model(img_path1)

    df = results1.pandas().xyxy[0]
    for index, row in df.iterrows():
        print(f"检测到: {row['name']} (ID={row['class']}), " f"置信度: {row['confidence']:.2f}, " f"位置: [{row['xmin']:.0f}, {row['ymin']:.0f}, {row['xmax']:.0f}, {row['ymax']:.0f}]")
    diff_tensor = get_yolo_diff4(results1, results)
    print(diff_tensor)
