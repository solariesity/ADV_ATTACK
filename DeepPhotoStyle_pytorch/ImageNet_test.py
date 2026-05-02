import torch
from torchvision import models, transforms
from PIL import Image
import os

# --------------------------
# 1. 基础配置（仅保留模型/图片相关，无类别名称映射）
# --------------------------
# 1.1 图片预处理（严格匹配ResNet输入格式，不可修改）
preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # 缩放至256x256
        transforms.CenterCrop(224),  # 中心裁剪至224x224（ResNet固定输入尺寸）
        transforms.ToTensor(),  # 转为Tensor（像素值0-1）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 用ImageNet均值/标准差标准化
    ]
)

# 1.2 设备自动配置（优先GPU，无则用CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用计算设备: {device}")


# --------------------------
# 2. 核心识别函数（仅返回预测的ImageNet index）
# --------------------------
def predict_imagenet_index(image_path, model):
    """
    识别图片对应的ImageNet目标编号（index，0-999）
    :param image_path: 待识别图片路径（如./test.jpg）
    :param model: 预训练ResNet模型
    :return: 预测结果（index、置信度、图片路径）
    """
    # 2.1 加载并预处理图片（处理灰度图/透明通道问题）
    try:
        image = Image.open(image_path).convert("RGB")  # 强制转为RGB格式
    except Exception as e:
        raise ValueError(f"图片加载失败: {str(e)}")

    # 预处理并增加batch维度（模型要求输入格式：[batch_size, 3, 224, 224]）
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 2.2 模型预测（关闭梯度计算，提升速度）
    model.eval()  # 切换为评估模式（禁用Dropout）
    with torch.no_grad():
        output = model(input_batch)  # 输出shape: [1, 1000]（对应1000个类别）

    # 2.3 解析结果：获取置信度最高的index
    probabilities = torch.nn.functional.softmax(output[0], dim=0)  # 转为0-1概率
    top1_prob, top1_idx = torch.max(probabilities, dim=0)  # 置信度最高的类别

    # 仅返回index（目标编号）及关键信息，无类别名称
    return {"ImageNet预测编号（index）": top1_idx.item(), "预测置信度": round(top1_prob.item(), 4), "待识别图片路径": image_path}  # 核心输出：0-999的编号  # 辅助信息：置信度（0-1）


# --------------------------
# 3. 运行示例（直接执行即可）
# --------------------------
if __name__ == "__main__":
    # 3.1 加载预训练模型（ResNet50，适配不同PyTorch版本）
    try:
        # 适配PyTorch < 1.13版本
        model = models.resnet50(pretrained=True)
    except AttributeError:
        # 适配PyTorch ≥ 1.13版本（pretrained参数废弃，改用weights）
        model = torch.hub.load("pytorch/vision:v0.14.1", "resnet50", weights="ResNet50_Weights.IMAGENET1K_V1")
    model = model.to(device)  # 模型送入指定设备（CPU/GPU）

    # 3.2 替换为你的待识别图片路径（支持JPG/PNG/BMP等常见格式）
    TEST_IMAGE_PATH = "/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/asset/src_img/car/BMW.png"  # 示例：如一张"金毛犬"图片（对应index=207）

    # 3.3 执行预测并打印结果
    try:
        result = predict_imagenet_index(TEST_IMAGE_PATH, model)
        print("\n=== ImageNet类别识别结果（仅输出index）===")
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"识别流程出错: {str(e)}")
