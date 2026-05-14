import os

import torch
from PIL import Image
import torchvision.transforms as transforms

import config
import my_utils as utils
from my_yolov5.yolov5_draw import plot_detections, plot_detections_official, preprocess_input_tensor_for_yolo, yolo_result_nms, yolo_result_nms_official
from my_yolov5.yolov5_model import import_yolov5s_model_1, import_yolov5s_model_2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_test")
IMAGE_PATH = os.path.join(CURRENT_DIR, "asset", "src_img", "car", "hongqi.jpg")


def round_up_to_stride(value, stride=32):
    return ((value + stride - 1) // stride) * stride


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, f"class_{class_id}")
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def print_tensor_detections(title, final_boxes, final_scores, final_class_ids, class_names):
    if len(final_scores) == 0:
        print(f"{title}: none above threshold.")
        return

    print(f"{title}:")
    for i, (box, score, class_id) in enumerate(zip(final_boxes, final_scores, final_class_ids)):
        x1, y1, x2, y2 = box.int().tolist()
        class_id_int = int(class_id.item())
        class_name = get_class_name(class_names, class_id_int)
        print(
            f"[{i}] class={class_name} class_id={class_id_int} "
            f"score={score.item():.6f} box=({x1}, {y1}, {x2}, {y2})"
        )

    best_idx = torch.argmax(final_scores).item()
    print(f"{title} best detection confidence: {final_scores[best_idx].item():.6f}")


def print_official_detections(title, detections, class_names):
    if detections is None or len(detections) == 0:
        print(f"{title}: none above threshold.")
        return

    print(f"{title}:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det.tolist()
        class_id = int(cls)
        class_name = get_class_name(class_names, class_id)
        print(
            f"[{i}] class={class_name} class_id={class_id} "
            f"score={conf:.6f} box=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
        )

    best_idx = int(torch.argmax(detections[:, 4]).item())
    print(f"{title} best detection confidence: {detections[best_idx, 4].item():.6f}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = Image.open(IMAGE_PATH).convert("RGB")
original_width, original_height = image.size
resized_width = round_up_to_stride(original_width, 32)
resized_height = round_up_to_stride(original_height, 32)
resized_image = transforms.Resize((resized_height, resized_width))(image)
input_tensor = utils.image_to_tensor(resized_image)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)

print(f"Loaded image: {IMAGE_PATH}")
print(f"Original size: {(original_width, original_height)}")
print(f"Resized size: {(resized_width, resized_height)}")
print(f"Tensor shape: {tuple(input_tensor.shape)}")
print(f"Device: {config.device0}")

device_num = config.device0.index if config.device0.type == "cuda" and config.device0.index is not None else 0
model_tensor = import_yolov5s_model_2(device_num, None, "yolov5s")
model_tensor.eval()

with torch.no_grad():
    model_output = model_tensor(input_tensor)
    output_img = plot_detections(model_output, input_tensor)
    final_boxes, final_scores, final_class_ids, class_names = yolo_result_nms(model_output, input_tensor)

output_path = os.path.join(OUTPUT_DIR, "hongqi_yolo_result_tensor.jpg")
output_img.save(output_path)
print(f"Saved tensor-path result image to: {output_path}")

print_tensor_detections("Tensor-path detections", final_boxes, final_scores, final_class_ids, class_names)

model_stride = getattr(model_tensor, "stride", 32)
if isinstance(model_stride, torch.Tensor):
    stride = int(max(model_stride.max().item(), 32))
elif isinstance(model_stride, (list, tuple)):
    stride = int(max(max(model_stride), 32))
else:
    stride = int(max(model_stride, 32))

full_tensor_input, _ = preprocess_input_tensor_for_yolo(input_tensor, image_size=(640, 640), stride=stride)
with torch.no_grad():
    full_model_output = model_tensor(full_tensor_input)

full_tensor_output, full_tensor_detections, full_tensor_names = plot_detections_official(
    full_model_output,
    input_tensor,
    image_size=(640, 640),
    stride=stride,
    conf_threshold=float(getattr(model_tensor, "conf", 0.1)),
    iou_threshold=float(getattr(model_tensor, "iou", 0.45)),
    class_names=getattr(model_tensor, "names", None),
    agnostic=bool(getattr(model_tensor, "agnostic", False)),
    max_det=int(getattr(model_tensor, "max_det", 300)),
)
full_tensor_output_path = os.path.join(OUTPUT_DIR, "hongqi_yolo_result_tensor_full.jpg")
full_tensor_output.save(full_tensor_output_path)
print(f"Saved full-tensor-path result image to: {full_tensor_output_path}")
print(f"Full tensor input shape: {tuple(full_tensor_input.shape)}")
print_official_detections("Full-tensor-path detections", full_tensor_detections, full_tensor_names)

model_image = import_yolov5s_model_1(device_num, None, "yolov5s")
model_image.eval()
results = model_image(IMAGE_PATH)

image_output = Image.fromarray(results.render()[0])
image_output_path = os.path.join(OUTPUT_DIR, "hongqi_yolo_result_import1.jpg")
image_output.save(image_output_path)
print(f"Saved import1-path result image to: {image_output_path}")

df = results.pandas().xyxy[0]
if df.empty:
    print("Import1-path detections: none above threshold.")
else:
    print("Import1-path detections:")
    for i, row in df.iterrows():
        print(
            f"[{i}] class={row['name']} class_id={int(row['class'])} "
            f"score={row['confidence']:.6f} "
            f"box=({int(row['xmin'])}, {int(row['ymin'])}, {int(row['xmax'])}, {int(row['ymax'])})"
        )

    best_row = df.loc[df["confidence"].idxmax()]
    print(f"Import1-path best detection confidence: {best_row['confidence']:.6f}")
