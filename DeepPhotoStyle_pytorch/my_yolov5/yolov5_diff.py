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
    基于 mask 的外接框筛选候选检测框。

    规则：
    1. 计算 mask 的最小外接框，并向四周扩展 50 像素；
    2. 仅保留完全落在该扩展框内部的检测框；
    3. 检测框面积至少占 mask 面积的 50%。

    参数:
        bbox: [x1, y1, x2, y2]（像素坐标）
        mask: [C, H, W] 或 [H, W] 的掩码
    返回:
        bool: 满足筛选条件时返回 True
    """
    x1, y1, x2, y2 = bbox.int().tolist()
    if mask.dim() == 3:
        mask_2d = mask.sum(dim=0) > 0
        h, w = mask.shape[1], mask.shape[2]
    else:
        mask_2d = mask > 0
        h, w = mask.shape

    ys, xs = torch.nonzero(mask_2d, as_tuple=True)
    if ys.numel() == 0 or xs.numel() == 0:
        return False

    mask_x1 = int(xs.min().item())
    mask_y1 = int(ys.min().item())
    mask_x2 = int(xs.max().item()) + 1
    mask_y2 = int(ys.max().item()) + 1

    margin = 50
    exp_x1 = max(0, mask_x1 - margin)
    exp_y1 = max(0, mask_y1 - margin)
    exp_x2 = min(w, mask_x2 + margin)
    exp_y2 = min(h, mask_y2 + margin)

    if x1 < exp_x1 or y1 < exp_y1 or x2 > exp_x2 or y2 > exp_y2:
        return False

    det_w = max(0, x2 - x1)
    det_h = max(0, y2 - y1)
    if det_w == 0 or det_h == 0:
        return False

    det_area = det_w * det_h
    mask_area = int(mask_2d.sum().item())
    if mask_area == 0:
        return False

    area_ratio = det_area / mask_area
    return area_ratio >= 0


def get_mask_filter_stats(mask, margin=50):
    """
    预计算与 mask 相关、且在候选框循环中保持不变的筛选信息。

    返回:
        dict | None:
        - exp_x1, exp_y1, exp_x2, exp_y2: 扩展框边界
        - mask_area: mask 像素面积
        若 mask 为空则返回 None
    """
    if mask.dim() == 3:
        mask_2d = mask.sum(dim=0) > 0
        h, w = mask.shape[1], mask.shape[2]
    else:
        mask_2d = mask > 0
        h, w = mask.shape

    ys, xs = torch.nonzero(mask_2d, as_tuple=True)
    if ys.numel() == 0 or xs.numel() == 0:
        return None

    mask_x1 = int(xs.min().item())
    mask_y1 = int(ys.min().item())
    mask_x2 = int(xs.max().item()) + 1
    mask_y2 = int(ys.max().item()) + 1

    exp_x1 = max(0, mask_x1 - margin)
    exp_y1 = max(0, mask_y1 - margin)
    exp_x2 = min(w, mask_x2 + margin)
    exp_y2 = min(h, mask_y2 + margin)

    mask_area = int(mask_2d.sum().item())
    if mask_area == 0:
        return None

    return {
        "exp_x1": exp_x1,
        "exp_y1": exp_y1,
        "exp_x2": exp_x2,
        "exp_y2": exp_y2,
        "mask_area": mask_area,
    }


def bbox_intersects_mask_with_stats(bbox, mask_stats):
    """
    使用预计算的 mask 统计信息筛选候选检测框。
    """
    x1, y1, x2, y2 = bbox.int().tolist()

    if x1 < mask_stats["exp_x1"] or y1 < mask_stats["exp_y1"] or x2 > mask_stats["exp_x2"] or y2 > mask_stats["exp_y2"]:
        return False

    det_w = max(0, x2 - x1)
    det_h = max(0, y2 - y1)
    if det_w == 0 or det_h == 0:
        return False

    det_area = det_w * det_h
    area_ratio = det_area / mask_stats["mask_area"]
    return area_ratio >= 0.5


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
    mask_stats = get_mask_filter_stats(mask)
    if mask_stats is None:
        zero = pred.sum() * 0.0
        return zero

    boxes = xyxy[batch_idx]  # [N, 4]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    det_w = (x2 - x1).clamp(min=0)
    det_h = (y2 - y1).clamp(min=0)
    det_area = det_w * det_h

    valid_mask = (
        (x1 >= mask_stats["exp_x1"])
        & (y1 >= mask_stats["exp_y1"])
        & (x2 <= mask_stats["exp_x2"])
        & (y2 <= mask_stats["exp_y2"])
        & (det_w > 0)
        & (det_h > 0)
        & ((det_area / mask_stats["mask_area"]) >= 0.5)
    )

    # -----------------------------
    # 3. 无有效目标时直接返回 0
    # -----------------------------
    if not valid_mask.any():
        zero = pred.sum() * 0.0
        return zero

    valid_scores = combined_score[batch_idx][valid_mask]
    valid_confs = conf[batch_idx][valid_mask]
    valid_class_probs = class_probs[batch_idx][valid_mask]
    valid_cls_idx = cls_idx[batch_idx][valid_mask]
    valid_boxes = boxes[valid_mask]
    valid_det_areas = det_area[valid_mask]

    # -----------------------------
    # 4. 选取得分最高的目标
    # -----------------------------
    best_idx = torch.argmax(valid_scores)
    best_conf = valid_confs[best_idx]
    best_class_prob = valid_class_probs[best_idx]
    # best_cls_idx = int(valid_cls_idx[best_idx].item())
    # best_bbox = valid_boxes[best_idx].detach().cpu().tolist()
    # best_area_ratio = valid_det_areas[best_idx] / mask_stats["mask_area"]

    # class_names = getattr(yolo_model, "names", None)
    # if isinstance(class_names, dict):
    #     best_class_name = class_names.get(best_cls_idx, str(best_cls_idx))
    # elif isinstance(class_names, (list, tuple)) and 0 <= best_cls_idx < len(class_names):
    #     best_class_name = class_names[best_cls_idx]
    # else:
    #     best_class_name = str(best_cls_idx)

    # print("best_class:", best_class_name)
    # print("best_bbox:", best_bbox)
    # print("best_area_ratio:", best_area_ratio.item())

    # -----------------------------
    # 5. 构造损失（可反向传播）
    # -----------------------------
    loss = torch.log10(1.0 - best_conf + 1e-6) - class_lambda * best_class_prob[2]

    return loss
