import os
from collections import Counter

import pynvml
import torch
from PIL import Image
from torchvision.ops import nms as nms_torchvision

import config


COCO_CLASS_NAMES = [
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


def print_tensor_info(tensor, name="张量"):
    """
    打印张量的详细统计信息。

    参数:
        tensor: 待打印信息的张量。
        name: 张量名称，仅用于日志展示。

    返回:
        无返回值。
    """
    print(f"=== {name} 信息 ===")
    print(f"形状: {tensor.shape}")
    print(f"维度数: {tensor.ndim}")
    print(f"数据类型: {tensor.dtype}")
    print(f"设备: {tensor.device}")
    print(f"值范围: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    print(f"均值: {tensor.mean().item():.3f}")
    print(f"标准差: {tensor.std().item():.3f}")
    print()


def print_config_gpu_memory():
    """
    打印 `config.device0` 对应 GPU 的显存占用信息。

    参数:
        无。

    返回:
        无返回值。
    """
    device = config.device0
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    print(f"{device} 内存占用:")
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已保留: {reserved:.2f} GB")
    print(f"  总内存: {total:.2f} GB")
    print(f"  使用率: {allocated / total * 100:.1f}%")


def non_max_suppression_torchvision(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    对 YOLO 输出执行基于 torchvision 的 NMS 后处理。

    参数:
        prediction: 模型输出，可以是列表或张量。
        conf_thres: 置信度阈值，低于该值的候选框会被过滤。
        iou_thres: NMS 的 IoU 阈值。

    返回:
        形状为 `[N, 6]` 的张量，格式为 `(x1, y1, x2, y2, confidence, class_id)`。
    """
    if isinstance(prediction, list):
        prediction_tensor = prediction[0]
    else:
        prediction_tensor = prediction

    if prediction_tensor.ndim == 3:
        prediction_tensor = prediction_tensor.squeeze(0)

    boxes = prediction_tensor[:, :4]
    obj_conf = prediction_tensor[:, 4]
    class_probs = prediction_tensor[:, 5:]

    class_scores, class_ids = torch.max(class_probs, dim=1)
    scores = obj_conf * class_scores

    valid_indices = (scores >= conf_thres).nonzero(as_tuple=True)[0]
    if len(valid_indices) == 0:
        return torch.empty((0, 6), device=prediction_tensor.device)

    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]

    x_center, y_center, width, height = boxes.unbind(1)
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    keep = nms_torchvision(boxes_xyxy, scores, iou_thres)
    result = torch.cat(
        [
            boxes_xyxy[keep],
            scores[keep].unsqueeze(1),
            class_ids[keep].unsqueeze(1).float(),
        ],
        dim=1,
    )
    return result


def print_detection_results(pred, class_names=None):
    """
    以可读表格形式打印检测结果。

    参数:
        pred: 检测结果张量，期望形状为 `[N, 6]`。
        class_names: 类别名称列表；为空时默认使用 COCO 类别。

    返回:
        无返回值。
    """
    if class_names is None:
        class_names = COCO_CLASS_NAMES

    print("=" * 100)
    print("检测结果详细输出")
    print("=" * 100)

    if not isinstance(pred, torch.Tensor):
        print(f"错误: pred 不是张量，而是 {type(pred)}")
        return

    print(f"pred 形状: {pred.shape}")
    print(f"pred 设备: {pred.device}")
    print(f"pred 数据类型: {pred.dtype}")
    print(f"元素总数: {pred.numel()}")

    if pred.numel() == 0:
        print("没有检测到任何目标")
        print("=" * 100)
        return

    if pred.ndim != 2 or pred.shape[1] != 6:
        print(f"警告: pred 形状异常 {pred.shape}，期望 [N, 6]")
        print("原始内容:")
        print(pred.tolist())
        print("=" * 100)
        return

    num_detections = pred.shape[0]
    print(f"检测到 {num_detections} 个目标")
    print("-" * 100)
    print(
        f"{'序号':<4} | {'类别ID':<6} | {'类别名称':<15} | {'置信度':<8} | "
        f"{'x1':<8} | {'y1':<8} | {'x2':<8} | {'y2':<8} | "
        f"{'宽度':<8} | {'高度':<8}"
    )
    print("-" * 100)

    for i in range(num_detections):
        detection = pred[i]
        try:
            if detection.numel() != 6:
                print(f"{i:<4} | 格式错误: {detection.tolist()}")
                continue

            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            class_id_int = int(round(class_id))

            if 0 <= class_id_int < len(class_names):
                class_name = class_names[class_id_int]
            else:
                class_name = f"unknown_{class_id_int}"

            width = x2 - x1
            height = y2 - y1

            print(
                f"{i:<4} | {class_id_int:<6} | {class_name:<15} | {confidence:.4f} | "
                f"{x1:8.2f} | {y1:8.2f} | {x2:8.2f} | {y2:8.2f} | "
                f"{width:8.2f} | {height:8.2f}"
            )
        except Exception as exc:
            print(f"{i:<4} | 解析错误: {exc}, 数据: {detection.tolist()}")

    print("-" * 100)

    confidences = pred[:, 4]
    class_ids = pred[:, 5].int()

    print(
        f"置信度统计: 平均={confidences.mean():.4f}, "
        f"最高={confidences.max():.4f}, 最低={confidences.min():.4f}"
    )
    print(
        f"边界框平均尺寸: 宽度={(pred[:, 2] - pred[:, 0]).mean():.2f}, "
        f"高度={(pred[:, 3] - pred[:, 1]).mean():.2f}"
    )

    class_counts = Counter(class_ids.tolist())
    print("类别分布:")
    for cls_id, count in class_counts.most_common():
        if 0 <= cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"unknown_{cls_id}"
        print(f"  {cls_name}: {count} 个")

    print("=" * 100)


def debug_pred_content(pred):
    """
    安全打印预测结果的基础结构信息，便于调试。

    参数:
        pred: 任意待检查对象，通常为模型输出或张量。

    返回:
        无返回值。
    """
    print("调试 pred 内容:")
    print(f"类型: {type(pred)}")

    if hasattr(pred, "shape"):
        print(f"形状: {pred.shape}")
    else:
        print("无 shape 属性")

    if hasattr(pred, "numel"):
        print(f"元素数量: {pred.numel()}")

    if hasattr(pred, "ndim"):
        print(f"维度: {pred.ndim}")

        if pred.ndim > 0:
            if pred.numel() > 0:
                print("前几个元素值:")
                if pred.ndim == 1:
                    print(pred[: min(10, pred.shape[0])].tolist())
                elif pred.ndim == 2:
                    print(pred[: min(5, pred.shape[0]), :].tolist())
            else:
                print("空张量")
    else:
        print("无法获取维度信息")

    print("-" * 50)


def print_memory_usage(message="1"):
    """
    打印当前 PyTorch 的显存占用信息。

    参数:
        message: 日志前缀，用于区分不同打印位置。

    返回:
        无返回值。
    """
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{message}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def print_gpu_usage():
    """
    使用 NVML 打印所有 GPU 的使用率、显存和温度信息。

    参数:
        无。

    返回:
        无返回值。
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = mem_info.total / 1024**3
            used_mem = mem_info.used / 1024**3
            free_mem = mem_info.free / 1024**3

            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory

            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            except Exception:
                temp = "N/A"

            print(f"GPU {i}: {name}")
            print(f"  GPU使用率: {gpu_util}%")
            print(f"  内存使用率: {memory_util}%")
            print(
                f"  内存: {used_mem:.2f}GB / {total_mem:.2f}GB "
                f"(空闲: {free_mem:.2f}GB)"
            )
            print(f"  温度: {temp}°C")
            print("-" * 50)
    except pynvml.NVMLError as exc:
        print(f"Error: {exc}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def tensor_to_jpg(tensor, file_path, data_range="0_1", channels_last=False):
    """
    将张量保存为 JPG 图片，支持单张和 batch 输入。

    参数:
        tensor: 输入张量，支持 `CHW`、`HWC`、`NCHW` 或 `NHWC`。
        file_path: 输出图片路径；如果是 batch，会自动追加索引。
        data_range: 张量取值范围，支持 `"0_1"` 或 `"-1_1"`。
        channels_last: 是否使用通道后置格式。

    返回:
        无返回值。
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    is_batch = tensor.ndim == 4
    if is_batch:
        tensors_to_process = tensor
        base_name, ext = os.path.splitext(file_path)
    else:
        tensors_to_process = [tensor]

    for index, current_tensor in enumerate(tensors_to_process):
        if not channels_last:
            current_tensor = current_tensor.permute(1, 2, 0)

        np_array = current_tensor.detach().numpy()

        if data_range == "-1_1":
            np_array = (np_array + 1.0) / 2.0

        np_array = np_array.clip(0, 1)
        np_array = (np_array * 255).astype("uint8")
        image = Image.fromarray(np_array)

        if is_batch:
            save_path = f"{base_name}_{index}{ext}"
        else:
            save_path = file_path

        image.save(save_path)
        print(f"图片已保存至: {save_path}")
