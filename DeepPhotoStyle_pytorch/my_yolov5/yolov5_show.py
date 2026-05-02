import os
import sys
import torch
from torchvision.ops import nms
import torch
import torchvision

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import config


# 辅助函数：边界框转换 (cxcywh to xyxy)
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_min
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_min
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_max
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_max
    return y


# 核心转换函数
def convert_result_2_to_detections(result_2, img_size=(1024, 320), conf_thres=0.25, iou_thres=0.45):
    """
    将YOLOv5原始输出转换为结构化检测结果
    Args:
        result_2: model_2的输出 (tuple)
        img_size: 模型输入尺寸 (width, height)
        conf_thres: 置信度阈值
        iou_thres: NMS的IoU阈值
    Returns:
        Detections对象 (类似result_1的结构)
    """
    # 提取预测张量 [batch_size, num_anchors, 85]
    pred = result_2[0]
    batch_size = pred.shape[0]
    device = pred.device

    # 解码参数设置 (基于YOLOv5s默认锚框)
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], device=device).float()  # P3/8  # P4/16  # P5/32

    # 缩放锚框到当前输入尺寸 (原始锚框基于640×640)
    scale_factor = torch.tensor([img_size[0] / 640, img_size[1] / 640], device=device)
    anchors = anchors * scale_factor

    # 特征图设置
    strides = [8, 16, 32]
    grid_sizes = [(img_size[1] // s, img_size[0] // s) for s in strides]
    num_anchors = anchors.shape[0] // 3

    # 处理每个批次图像
    detections = []
    for b in range(batch_size):
        # === 1. 边界框解码 ===
        batch_pred = pred[b]  # [20160, 85]

        # 按特征层级分离预测
        start_idx = 0
        decoded_preds = []
        for i, (h, w) in enumerate(grid_sizes):
            end_idx = start_idx + h * w * num_anchors
            layer_pred = batch_pred[start_idx:end_idx]
            layer_pred = layer_pred.view(h, w, num_anchors, -1)  # [H, W, A, 85]

            # 生成网格坐标
            grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1).float()  # [H, W, 2]

            # 获取当前层级的锚框
            anchor_idx = i * num_anchors
            layer_anchors = anchors[anchor_idx : anchor_idx + num_anchors].view(1, 1, num_anchors, 2)

            # 解码边界框 (YOLOv5 v6.0+ 解码公式)
            xy = (torch.sigmoid(layer_pred[..., 0:2]) * 2 - 0.5 + grid[:, :, None, :]) * strides[i]
            wh = (torch.sigmoid(layer_pred[..., 2:4]) * 2) ** 2 * layer_anchors
            box = torch.cat((xy, wh), dim=-1)  # [cx, cy, w, h]

            # 转换格式: [x_min, y_min, x_max, y_max]
            box = xywh2xyxy(box.view(-1, 4))

            # 置信度和类别
            conf = torch.sigmoid(layer_pred[..., 4:5]).view(-1, 1)
            cls = torch.sigmoid(layer_pred[..., 5:]).view(-1, 80)

            # 保存解码结果
            decoded_layer = torch.cat([box, conf, cls], dim=-1)
            decoded_preds.append(decoded_layer)
            start_idx = end_idx

        # 合并所有层级的预测
        all_preds = torch.cat(decoded_preds, dim=0)  # [20160, 85]

        # === 2. 置信度过滤 ===
        # 计算最大类别分数
        max_scores, max_classes = torch.max(all_preds[:, 5:], dim=1)
        valid_mask = (all_preds[:, 4] * max_scores) > conf_thres

        # 过滤低置信度预测
        filtered_preds = all_preds[valid_mask]
        max_scores = max_scores[valid_mask]
        max_classes = max_classes[valid_mask]

        # 组合为 [x_min, y_min, x_max, y_max, conf, class_id]
        batch_detections = torch.cat([filtered_preds[:, :4], (filtered_preds[:, 4] * max_scores).unsqueeze(1), max_classes.unsqueeze(1).float()], dim=1)

        # === 3. 执行NMS ===
        if batch_detections.shape[0] > 0:
            keep = torchvision.ops.nms(batch_detections[:, :4], batch_detections[:, 4], iou_thres)  # 边界框  # 置信度
            batch_detections = batch_detections[keep]

        detections.append(batch_detections)

    # === 4. 构建结构化结果 ===
    class Detections:
        def __init__(self, detections, img_size):
            self.pred = detections
            self.xyxy = detections
            self.n = len(detections)
            self.s = (batch_size, 3) + img_size[::-1]  # (batch, channels, height, width)

            # 其他格式的坐标 (此处简化实现，实际使用时可计算)
            self.xywh = [xyxy2xywh(d[:, :4]) for d in detections]
            self.xyxyn = [d[:, :4] / torch.tensor([img_size[0], img_size[1], img_size[0], img_size[1]], device=device) for d in detections]
            self.xywhn = [xyxy2xywh(d) / torch.tensor([img_size[0], img_size[1], img_size[0], img_size[1]], device=device) for d in self.xyxyn]

            # COCO类别名称
            self.names = {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                4: "airplane",
                5: "bus",
                6: "train",
                7: "truck",
                8: "boat",
                9: "traffic light",
                # ... 完整列表参考COCO类别 ...
                79: "toothbrush",
            }

    return Detections(detections, img_size)


def print_detection_structure(detection, indent=0):
    """
    autoshape=False:
    list with 2 elements:
    [0]:
        Tensor: shape=torch.Size([6, 20160, 85]), dtype=torch.float32, device=cuda:0
        Values range: min=0.0000, max=1021.3836
    [1]:
        list with 3 elements:
        [0]:
            Tensor: shape=torch.Size([6, 3, 40, 128, 85]), dtype=torch.float32, device=cuda:0
            Values range: min=-18.0400, max=3.7849
        [1]:
            Tensor: shape=torch.Size([6, 3, 20, 64, 85]), dtype=torch.float32, device=cuda:0
            Values range: min=-13.4076, max=2.8083
        [2]:
            Tensor: shape=torch.Size([6, 3, 10, 32, 85]), dtype=torch.float32, device=cuda:0
            Values range: min=-13.6573, max=3.8527

    """
    prefix = "  " * indent
    if isinstance(detection, torch.Tensor):
        print(f"{prefix}Tensor: shape={detection.shape}, dtype={detection.dtype}, device={detection.device}")
        if detection.numel() > 0:
            print(f"{prefix}  Values range: min={detection.min().item():.4f}, max={detection.max().item():.4f}")

    elif isinstance(detection, (list, tuple)):
        print(f"{prefix}{type(detection).__name__} with {len(detection)} elements:")
        for i, item in enumerate(detection):
            print(f"{prefix}  [{i}]:")
            print_detection_structure(item, indent + 2)

    elif hasattr(detection, "__dict__") or hasattr(detection, "__slots__"):
        print(f"{prefix}Object of type {type(detection).__name__}:")
        if isinstance(detection, dict):
            for key, value in detection.items():
                print(f"{prefix}  {key}:")
                print_detection_structure(value, indent + 2)
        else:
            # 尝试获取对象的公共属性
            attrs = vars(detection)
            for key, value in attrs.items():
                print(f"{prefix}  {key}:")
                print_detection_structure(value, indent + 2)

    else:
        print(f"{prefix}{type(detection).__name__}: {str(detection)[:100]}{'...' if len(str(detection)) > 100 else ''}")


# from my_yolov5.yolov5_model import import_yolov5s_model, import_yolov5s_model_2

if __name__ == "__main__":
    random_tensor = torch.randn(6, 3, 320, 1024).to(config.device0)
    model_2 = torch.hub.load("ultralytics/yolov5", "custom", path="../models/yolov5s.pt", autoshape=False).to(config.device0)  # 使用path参数指定模型路径
    model_2.eval()
    detection = model_2(random_tensor)
