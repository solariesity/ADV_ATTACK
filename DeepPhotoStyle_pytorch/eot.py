import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math


def apply_random_transforms(scene_img: torch.Tensor) -> torch.Tensor:
    """
    EOT (Expectation over Transformation) 模块：
    对输入图像施加随机颜色扰动、几何变换、模糊与噪声，
    用于增强模型在物理扰动下的鲁棒性。

    Args:
        scene_img (Tensor): [B, C, H, W], range in [0, 1]

    Returns:
        Tensor: 变换后的图像，形状不变
    """
    B, C, H, W = scene_img.shape
    device = scene_img.device

    x = scene_img.clone()

    # =====================================================
    # 1. 颜色扰动（Brightness / Contrast / Saturation）
    # =====================================================
    brightness = 1.0 + (torch.rand(B, device=device) - 0.5) * 0.4
    contrast = 1.0 + (torch.rand(B, device=device) - 0.5) * 0.4
    saturation = 1.0 + (torch.rand(B, device=device) - 0.5) * 0.4

    for i in range(B):
        x[i] = TF.adjust_brightness(x[i], brightness[i].item())
        x[i] = TF.adjust_contrast(x[i], contrast[i].item())
        x[i] = TF.adjust_saturation(x[i], saturation[i].item())

    # =====================================================
    # 2. 随机仿射变换（旋转 + 缩放）
    # =====================================================
    max_rotate = 10.0  # degree
    angles = (torch.rand(B, device=device) * 2 - 1) * max_rotate
    angles = angles * math.pi / 180.0

    scales = torch.rand(B, device=device) * (1.1 - 0.9) + 0.9

    theta = torch.zeros(B, 2, 3, device=device)

    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales

    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    # 平移项置 0（可扩展）

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")

    # =====================================================
    # 3. 高斯模糊（batch 级随机）
    # =====================================================
    if torch.rand(1, device=device) > 0.5:
        k = torch.randint(3, 8, (1,), device=device).item()
        if k % 2 == 0:
            k += 1
        sigma = torch.rand(1, device=device).item() * 1.5 + 0.1
        x = TF.gaussian_blur(x, kernel_size=k, sigma=[sigma, sigma])

    # =====================================================
    # 4. 加性高斯噪声
    # =====================================================
    if torch.rand(1, device=device) > 0.7:
        noise = torch.randn_like(x) * 0.02
        x = x + noise

    # =====================================================
    # 5. 裁剪到合法范围
    # =====================================================
    x = torch.clamp(x, 0.0, 1.0)

    return x
