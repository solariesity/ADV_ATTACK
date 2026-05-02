import sys
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

try:
    from .yolov5_draw import *
    from .yolov5_show import *
except ImportError:
    from yolov5_draw import *
    from yolov5_show import *


def batch_xywh2xyxy(xywh):
    """
    将形状为 [..., 4] 的张量从 xywh 格式转换为 xyxy 格式。
    参数:
        xywh (torch.Tensor): 输入张量，形状为 [..., 4]，每个元素为 (x_center, y_center, width, height)。
    返回:
        torch.Tensor: 转换后的张量，形状与输入相同，每个元素为 (x1, y1, x2, y2)。
    """
    xyxy = xywh.clone()  # 创建副本以避免修改原始张量
    # 计算 x1, y1, x2, y2
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2  # x1 = x_center - width/2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2  # y1 = y_center - height/2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2  # x2 = x_center + width/2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2  # y2 = y_center + height/2
    return xyxy


def bbox_intersects_mask(bbox, mask):
    """
    检查边界框扩大10像素后是否与掩码有交集，并检测边界是否全为0
    参数:
        bbox: [x1, y1, x2, y2]（像素坐标）
        mask: [C, H, W]的掩码
    返回:
        bool: 当且仅当满足以下条件时返回True:
              a) 扩大后的区域与掩码有交集
              b) 扩大后的区域边界全为0
    """
    x1, y1, x2, y2 = bbox.int().tolist()
    h, w = mask.shape[1], mask.shape[2]

    # 扩大10像素（确保不超出图像边界）
    x1_exp = max(0, x1 - 50)
    y1_exp = max(0, y1 - 50)
    x2_exp = min(w, x2 + 50)
    y2_exp = min(h, y2 + 50)

    mask_area = mask[:, y1_exp:y2_exp, x1_exp:x2_exp]

    # 检查是否有交集
    if mask_area.sum() == 0:
        return False

    # # 检查边界是否有1（仅当区域非空时）
    # if mask_area.numel() == 0:
    #     return True

    # 检查四条边界
    has_boundary_1 = mask_area[:, 0, :].sum() > 0 or mask_area[:, -1, :].sum() > 0 or mask_area[:, :, 0].sum() > 0 or mask_area[:, :, -1].sum() > 0  # 上边界  # 下边界  # 左边界  # 右边界
    return not has_boundary_1


def resize_mask(mask: torch.Tensor, target_size: Tuple[int, int] = (640, 640), fill_value: int = 0) -> Tuple[torch.Tensor, List[Dict]]:
    """
    使用与YOLOv5图像预处理相同的方式处理mask张量

    参数:
        mask: 输入mask张量，形状为 [B, C, H, W]
        target_size: 目标尺寸 (height, width)，默认 (640, 640)
        fill_value: 填充值，默认0（表示背景/忽略区域）

    返回:
        tuple: (预处理后的mask, 转换参数列表)
        - 预处理后的mask: 形状为 [B, C, target_size[0], target_size[1]]
        - 转换参数列表: 每个元素包含单张mask的转换参数
    """
    # 确保输入是4D张量
    if mask.ndim != 4:
        raise ValueError(f"输入mask应为4D [B, C, H, W]，但得到 {mask.shape}")

    batch_size = mask.shape[0]
    processed_masks = []
    transform_params_list = []

    # 遍历batch中的每个mask
    for i in range(batch_size):
        single_mask = mask[i]  # [C, H, W]

        # 获取原始尺寸
        _, h, w = single_mask.shape
        target_h, target_w = target_size

        # 计算缩放比例（保持长宽比）
        scale = min(target_w / w, target_h / h)

        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 使用最近邻插值缩放mask（保持离散值）
        resized_mask = F.interpolate(
            single_mask.unsqueeze(0),  # 添加batch维度 [1, C, H, W]
            size=(new_h, new_w),
            mode="nearest",
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

        # 添加填充（使用fill_value）
        padded_mask = F.pad(
            resized_mask,
            (left, right, top, bottom),  # 填充顺序: 左, 右, 上, 下
            mode="constant",
            value=fill_value,
        )

        processed_masks.append(padded_mask)

        # 保存当前mask的转换参数
        transform_params = {
            "orig_size": (h, w),  # 原始尺寸 (H, W)
            "target_size": (target_h, target_w),  # 目标尺寸 (H, W)
            "scale": scale,
            "padding": (left, top, right, bottom),  # 填充 (left, top, right, bottom)
            "new_size": (new_h, new_w),  # 缩放后尺寸 (H, W)
        }
        transform_params_list.append(transform_params)

    # 将处理后的mask堆叠成batch
    batch_tensor = torch.stack(processed_masks)  # [B, C, H, W]

    # 打印resize后mask的属性（以第一个mask为例）
    # print("\n=== Resize后的Mask属性 ===")
    # print(f"形状: {batch_tensor.shape}")
    # print(f"数据类型: {batch_tensor.dtype}")
    # print(f"设备: {batch_tensor.device}")
    # print(f"最小值: {batch_tensor.min().item()}")
    # print(f"最大值: {batch_tensor.max().item()}")
    # print(f"均值: {batch_tensor.float().mean().item():.4f}")
    # print(f"非零元素比例: {(batch_tensor != 0).float().mean().item() * 100:.2f}%")
    # print("========================\n")

    return batch_tensor


def get_yolo_diff(
    adv_yolo,
    car_yolo,
    scene_obj_mask,
    yolo_model,
    class_lambda,
    class_weight=0.0,
    bbox_weight=0.0,
    conf_weight=5.0,
):
    """
    计算 YOLO 输出在给定场景掩码下的差异值（支持反向传播）

    参数:
        adv_yolo: YOLO 对抗样本输出
        car_yolo: 原始样本输出（当前未使用，可扩展）
        scene_obj_mask: [B, H, W]，目标区域掩码
        class_lambda: 类别约束权重
        class_weight: 预留
        bbox_weight: 预留
        conf_weight: 置信度权重

    返回:
        标量 loss（可反向传播）
    """

    # -----------------------------
    # 1. 解析 YOLO 输出
    # -----------------------------
    pred = adv_yolo[0]  # [B, N, 5 + num_classes]
    conf = pred[..., 4]  # [B, N]
    xywh = pred[..., :4]  # [B, N, 4]
    xyxy = batch_xywh2xyxy(xywh)  # [B, N, 4]

    class_probs = torch.softmax(pred[..., 5:], dim=-1)
    max_cls_prob, cls_idx = torch.max(class_probs, dim=-1)

    # 置信度 × 类别概率
    combined_score = conf * max_cls_prob

    # -----------------------------
    # 2. 只保留 mask 内的候选框
    # -----------------------------
    batch_idx = 0
    mask = scene_obj_mask[batch_idx]

    valid_scores = []
    valid_confs = []
    valid_class_probs = []

    for i in range(pred.shape[1]):
        if not bbox_intersects_mask(xyxy[batch_idx, i], mask):
            continue

        valid_scores.append(combined_score[batch_idx, i])
        valid_confs.append(conf[batch_idx, i])
        valid_class_probs.append(class_probs[batch_idx, i])

    # -----------------------------
    # 3. 无有效目标时直接返回 0
    # -----------------------------
    if len(valid_scores) == 0:
        return torch.tensor(0.0, device=pred.device)

    valid_scores = torch.stack(valid_scores)
    valid_confs = torch.stack(valid_confs)
    valid_class_probs = torch.stack(valid_class_probs)

    # -----------------------------
    # 4. 选取得分最高的目标
    # -----------------------------
    best_idx = torch.argmax(valid_scores)
    best_conf = valid_confs[best_idx]
    best_class_prob = valid_class_probs[best_idx]

    # -----------------------------
    # 5. 构造损失（可反向传播）
    # -----------------------------
    loss = torch.log10(1.0 - best_conf + 1e-6) - class_lambda * best_class_prob[2]

    return loss
