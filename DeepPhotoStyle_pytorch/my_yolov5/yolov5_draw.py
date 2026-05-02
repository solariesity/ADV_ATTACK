import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.ops as ops
from typing import Dict, List, Tuple, Union
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import config


def tensor_preprocess_for_yolov5(
    tensor: torch.Tensor,
    target_size: Tuple[int, int] = (640, 640),
    fill_color: float = 114.0 / 255.0,  # 归一化后的填充值
    device: Union[str, torch.device] = "cuda",
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    将输入张量预处理为YOLOv5模型所需的格式（可微版本）

    参数:
        tensor: 输入张量，形状为 [B, C, H, W]，值范围 [0, 1]
        target_size: 目标尺寸 (height, width)，默认 (640, 640)
        fill_color: 填充颜色（归一化后的值），默认 114/255
        device: 输出张量设备，默认 'cuda'

    返回:
        tuple: (预处理后的张量, 转换参数列表)
        - 预处理后的张量: 形状为 [B, 3, target_size[0], target_size[1]]，值范围 [0, 1]
        - 转换参数列表: 每个元素包含单张图像的转换参数
    """
    # 确保输入是4D张量 [B, C, H, W]
    if tensor.ndim != 4:
        raise ValueError(f"输入张量应为4D [B, C, H, W]，但得到 {tensor.shape}")

    batch_size = tensor.shape[0]
    processed_images = []
    transform_params_list = []

    # 遍历batch中的每张图像
    for i in range(batch_size):
        img_tensor = tensor[i]  # [C, H, W]

        # 获取原始尺寸
        _, h, w = img_tensor.shape
        target_h, target_w = target_size

        # 计算缩放比例（保持长宽比）
        scale = min(target_w / w, target_h / h)

        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 使用双线性插值缩放图像（可微操作）
        resized_img = F.interpolate(
            img_tensor.unsqueeze(0),  # 添加batch维度 [1, C, H, W]
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # 移除batch维度 [C, new_h, new_w]

        # 计算填充大小
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        # 分配填充（上下左右均分）
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # 创建填充后的张量（可微操作）
        # 方法1: 使用F.pad
        padded_img = F.pad(
            resized_img,
            (left, right, top, bottom),  # 填充顺序: 左, 右, 上, 下
            mode="constant",
            value=fill_color,
        )

        # 方法2: 使用切片赋值（更直观，但需要确保内存连续）
        # padded_img = torch.full((img_tensor.shape[0], target_h, target_w),
        #                       fill_color, device=img_tensor.device, dtype=img_tensor.dtype)
        # padded_img[:, top:top+new_h, left:left+new_w] = resized_img

        processed_images.append(padded_img)

        # 保存当前图像的转换参数
        transform_params = {
            "orig_size": (h, w),  # 原始尺寸 (H, W)
            "target_size": (target_h, target_w),  # 目标尺寸 (H, W)
            "scale": scale,
            "padding": (left, top, right, bottom),  # 填充 (left, top, right, bottom)
            "new_size": (new_h, new_w),  # 缩放后尺寸 (H, W)
        }
        transform_params_list.append(transform_params)

    # 将处理后的图像堆叠成batch并移动到指定设备
    batch_tensor = torch.stack(processed_images).to(config.device0)  # [B, C, H, W]

    return batch_tensor


def print_xywh_elements(model_output, input_name, num_elements=10000):
    """
    打印模型输出中的xywh元素

    参数:
        model_output: YOLO模型输出
        input_name: 输入名称（用于打印）
        num_elements: 要打印的元素数量
    """
    print(f"\n=== {input_name} 的xywh元素 ===")

    # 获取模型输出
    raw_predictions = model_output[0]  # 形状: [batch_size, num_anchors, 85]

    # 取第一张图像的预测结果
    predictions = raw_predictions[0]  # [num_anchors, 85]

    # 分离边界框、置信度和类别概率
    boxes_xywh = predictions[:, :4]  # [x_center, y_center, width, height] (绝对坐标)
    obj_conf = predictions[:, 4]  # 对象置信度
    class_probs = predictions[:, 5:]  # 类别概率

    # 计算每个检测的类别置信度
    class_scores, class_ids = torch.max(class_probs, dim=1)
    final_scores = obj_conf * class_scores

    # 打印前num_elements个检测框的xywh值
    print(f"总检测框数量: {len(boxes_xywh)}")
    print(f"前{num_elements}个检测框的xywh值:")

    for i in range(min(num_elements, len(boxes_xywh))):
        x, y, w, h = boxes_xywh[i]
        score = final_scores[i]
        class_id = class_ids[i]
        if score > 0.5:
            print(f"  检测框 {i+1}: xywh=({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), 置信度={score:.4f}, 类别={class_id.item()}")


def yolo_result_nms(model_output, input_tensor, conf_threshold=0.25, iou_threshold=0.45, class_names=None):
    """
    处理YOLO检测结果，应用NMS并返回处理后的检测信息

    参数:
        model_output: YOLO模型输出，格式为 [raw_predictions, ...]
        input_tensor: 输入图像张量，形状为 [batch_size, 3, height, width]
        conf_threshold: 置信度阈值，默认0.25
        iou_threshold: NMS的IOU阈值，默认0.45
        class_names: 类别名称列表，如果为None则使用COCO类别

    返回:
        tuple: (final_boxes, final_scores, final_class_ids, class_names)
            final_boxes: NMS后的边界框 [N, 4] (xyxy格式)
            final_scores: NMS后的置信度 [N]
            final_class_ids: NMS后的类别ID [N]
            class_names: 类别名称列表
    """
    # 获取图像尺寸
    img_height, img_width = input_tensor.shape[2], input_tensor.shape[3]
    # print(f"图像尺寸: {img_width}x{img_height}")

    # 获取模型输出
    raw_predictions = model_output[0]  # 形状: [batch_size, num_anchors, 85]

    # 取第一张图像的预测结果
    predictions = raw_predictions[0]  # [num_anchors, 85]

    # 分离边界框、置信度和类别概率
    # YOLO输出的是绝对坐标 [x_center, y_center, width, height]
    boxes = predictions[:, :4]  # [x_center, y_center, width, height] (绝对坐标)
    obj_conf = predictions[:, 4]  # 对象置信度
    class_probs = predictions[:, 5:]  # 类别概率

    # 计算每个检测的类别置信度
    class_scores, class_ids = torch.max(class_probs, dim=1)
    final_scores = obj_conf * class_scores

    # 过滤低置信度的检测
    mask = final_scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = final_scores[mask]
    filtered_class_ids = class_ids[mask]

    # print(f"过滤后剩余检测数量: {len(filtered_boxes)}")

    if len(filtered_boxes) == 0:
        # print("没有检测到任何物体")
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), class_names

    # 转换边界框格式：从 [x_center, y_center, width, height] 到 [x1, y1, x2, y2]
    # 由于已经是绝对坐标，直接转换即可
    boxes_xyxy = torch.zeros_like(filtered_boxes)
    boxes_xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1 = x_center - width/2
    boxes_xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1 = y_center - height/2
    boxes_xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2 = x_center + width/2
    boxes_xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2 = y_center + height/2

    # 确保坐标在图像范围内
    boxes_xyxy[:, 0] = torch.clamp(boxes_xyxy[:, 0], 0, img_width - 1)
    boxes_xyxy[:, 1] = torch.clamp(boxes_xyxy[:, 1], 0, img_height - 1)
    boxes_xyxy[:, 2] = torch.clamp(boxes_xyxy[:, 2], 0, img_width - 1)
    boxes_xyxy[:, 3] = torch.clamp(boxes_xyxy[:, 3], 0, img_height - 1)

    # 应用NMS
    keep_indices = ops.nms(boxes_xyxy, filtered_scores, iou_threshold)

    # 获取NMS后的结果
    final_boxes = boxes_xyxy[keep_indices]
    final_scores = filtered_scores[keep_indices]
    final_class_ids = filtered_class_ids[keep_indices]

    # 设置默认的COCO类别名称（如果未提供）
    if class_names is None:
        class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    # 打印检测结果
    # print(f"\n=== NMS后的检测结果（共{len(final_boxes)}个） ===")
    for i, (box, score, class_id) in enumerate(zip(final_boxes, final_scores, final_class_ids)):
        x1, y1, x2, y2 = box.int().tolist()
        width = x2 - x1
        height = y2 - y1
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"

        # print(f"检测 {i}: 类别={class_name}, 置信度={score:.3f}, 坐标=({x1},{y1},{x2},{y2}), 宽={width}, 高={height}")

    return final_boxes, final_scores, final_class_ids, class_names


def plot_detections(model_output, input_tensor, conf_threshold=0.25, iou_threshold=0.45, class_names=None):
    """
    在输入图像上绘制YOLO检测结果，调用process_detections处理检测结果

    参数:
        model_output: YOLO模型输出，格式为 [raw_predictions, ...]
        input_tensor: 输入图像张量，形状为 [batch_size, 3, height, width]
        conf_threshold: 置信度阈值，默认0.25
        iou_threshold: NMS的IOU阈值，默认0.45
        class_names: 类别名称列表，如果为None则使用COCO类别

    返回:
        PIL Image对象，包含绘制了检测框的图像
    """
    # 获取第一张图像
    img_tensor = input_tensor[0]  # [3, height, width]

    # 将张量转换为PIL图像
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()  # [height, width, 3]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    image = Image.fromarray(img_np)

    # 获取图像尺寸
    img_height, img_width = image.size[1], image.size[0]

    # 调用process_detections处理检测结果
    final_boxes, final_scores, final_class_ids, class_names = yolo_result_nms(model_output, input_tensor, conf_threshold, iou_threshold, class_names)

    # 如果没有检测到任何物体，直接返回原图
    if len(final_boxes) == 0:
        return image

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    # 定义颜色映射
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (255, 128, 0),
        (128, 255, 0),
        (0, 128, 255),
    ]

    # 绘制每个检测框
    for i, (box, score, class_id) in enumerate(zip(final_boxes, final_scores, final_class_ids)):
        # 转换为整数坐标
        x1, y1, x2, y2 = box.int().tolist()

        # 确保坐标在图像范围内（双重检查）
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        # 确保x1 < x2且y1 < y2
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1

        # 选择颜色
        color = colors[int(class_id) % len(colors)]

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 准备标签文本
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
        label = f"{class_name} {score:.2f}"

        # 计算文本大小
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            # 如果textbbox不可用，使用textsize
            text_width, text_height = draw.textsize(label, font=font)

        # 绘制标签背景（在边界框上方）
        label_x1 = x1
        label_y1 = max(0, y1 - text_height - 4)
        label_x2 = x1 + text_width + 4
        label_y2 = y1

        # 确保标签在图像范围内
        label_x1 = max(0, label_x1)
        label_y1 = max(0, label_y1)
        label_x2 = min(img_width, label_x2)
        label_y2 = min(img_height, label_y2)

        # 只有当标签区域有效时才绘制
        if label_x2 > label_x1 and label_y2 > label_y1:
            draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
            draw.text((label_x1 + 2, label_y1), label, fill="white", font=font)

    return image


def convert_xywh_to_original(model_output, conf_threshold=0.25):
    """
    将模型在(640,640)图像上检测到的边界框(xywh格式)转换回原始(320,1024)图像上的边界框(xywh格式)
    保持返回值的类型与model_output相同

    参数:
        model_output: YOLO模型输出，格式为 [raw_predictions, ...]
        conf_threshold: 置信度阈值，默认0.25

    返回:
        与model_output相同格式的列表，其中边界框坐标已转换到原始图像尺寸
    """
    # 创建输出列表，复制非张量元素
    converted_output = []
    for item in model_output:
        if isinstance(item, torch.Tensor):
            converted_output.append(item.clone())
        else:
            converted_output.append(item)

    # 获取预测张量（假设是第一个元素）
    raw_predictions = converted_output[0]  # 形状: [batch_size, num_anchors, 85]

    # 处理batch中的每张图像
    for batch_idx in range(raw_predictions.shape[0]):
        predictions = raw_predictions[batch_idx]  # [num_anchors, 85]

        # 分离边界框、对象置信度和类别概率
        boxes_xywh = predictions[:, :4]  # [x_center, y_center, width, height] (在640x640图像上)
        obj_conf = predictions[:, 4]  # 对象置信度
        class_probs = predictions[:, 5:]  # 类别概率

        # 计算每个检测的类别置信度
        class_scores, class_ids = torch.max(class_probs, dim=1)
        final_scores = obj_conf * class_scores

        # 转换参数（根据预处理过程计算）
        orig_h, orig_w = 320, 1024  # 原始图像尺寸
        target_h, target_w = 640, 640  # 预处理后图像尺寸
        scale = min(target_w / orig_w, target_h / orig_h)  # 0.625
        new_w = int(orig_w * scale)  # 640
        new_h = int(orig_h * scale)  # 200
        left = (target_w - new_w) // 2  # 0
        top = (target_h - new_h) // 2  # 220

        # 转换每个边界框
        for i in range(len(boxes_xywh)):
            x, y, w, h = boxes_xywh[i]

            # 1. 减去填充区域
            x_eff = x - left
            y_eff = y - top

            # 2. 检查边界框是否在有效区域内
            x1_eff = x_eff - w / 2
            y1_eff = y_eff - h / 2
            x2_eff = x_eff + w / 2
            y2_eff = y_eff + h / 2

            # 如果边界框部分或全部在填充区域，则将对象置信度设为0
            if x1_eff < 0 or y1_eff < 0 or x2_eff > new_w or y2_eff > new_h:
                predictions[i, 4] = 0  # 将对象置信度设为0
                continue

            # 3. 缩放回原始图像尺寸
            x_orig = x_eff / scale
            y_orig = y_eff / scale
            w_orig = w / scale
            h_orig = h / scale

            # 确保坐标在原始图像范围内
            x_orig = max(w_orig / 2, min(x_orig, orig_w - w_orig / 2))
            y_orig = max(h_orig / 2, min(y_orig, orig_h - h_orig / 2))
            w_orig = min(w_orig, orig_w)
            h_orig = min(h_orig, orig_h)

            # 更新边界框坐标
            predictions[i, 0] = x_orig
            predictions[i, 1] = y_orig
            predictions[i, 2] = w_orig
            predictions[i, 3] = h_orig

    return converted_output
