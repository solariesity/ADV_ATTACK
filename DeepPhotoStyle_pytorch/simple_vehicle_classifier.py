import torch
import numpy as np
from PIL import Image


def simple_vehicle_classification(cam_model, image_with_car, image_scene_only, target_indices):
    """
    简单但实用的车辆分类方法
    通过对比有车和无车场景来确定真正的车辆类型
    """

    print("=== 简单车辆分类分析 ===")
    print("正在分析图像中的车辆类型...")

    # 存储结果
    vehicle_scores = {}
    scene_scores = {}

    # 分析所有类别
    for i, target_idx in enumerate(target_indices):
        # 有车场景的激活
        mask_car, pred_car = cam_model(image_with_car, i, "./")
        activation_car = torch.sum(torch.relu(mask_car)).item()

        # 纯场景的激活
        mask_scene, pred_scene = cam_model(image_scene_only, i, "./")
        activation_scene = torch.sum(torch.relu(mask_scene)).item()

        vehicle_scores[target_idx] = activation_car
        scene_scores[target_idx] = activation_scene

    # 计算车辆特异性得分
    vehicle_specific_scores = {}
    for target_idx in target_indices:
        car_score = vehicle_scores[target_idx]
        scene_score = scene_scores[target_idx]

        # 车辆特异性 = 有车激活 - 纯场景激活
        specificity = car_score - scene_score

        # 相对提升 = (有车激活 - 纯场景激活) / 纯场景激活
        if scene_score > 0:
            relative_boost = specificity / scene_score
        else:
            relative_boost = float("inf") if specificity > 0 else 0

        vehicle_specific_scores[target_idx] = {"car_activation": car_score, "scene_activation": scene_score, "specificity": specificity, "relative_boost": relative_boost}

    # 类别名称映射
    class_names = {
        407: "ambulance",
        428: "barrow",
        436: "beach wagon",
        450: "bobsled",
        468: "cab",
        511: "convertible",
        537: "dogsled",
        554: "fireboat",
        555: "fire engine",
        561: "forklift",
        565: "freight car",
        569: "garbage truck",
        573: "go-kart",
        575: "golfcart",
        586: "half track",
        595: "harvester",
        603: "horse cart",
        609: "jeep",
        627: "limousine",
        654: "minibus",
        656: "minivan",
        660: "mobile home",
        661: "Model T",
        665: "moped",
        670: "motor scooter",
        671: "mountain bike",
        675: "moving van",
        690: "oxcart",
        705: "passenger car",
        717: "pickup",
        734: "police van",
        751: "racer",
        757: "recreational vehicle",
        779: "school bus",
        791: "shopping cart",
        802: "snowmobile",
        803: "snowplow",
        817: "sports car",
        829: "streetcar",
        847: "tank",
        864: "tow truck",
        866: "tractor",
        867: "trailer truck",
        870: "tricycle",
        874: "trolleybus",
    }

    # 排序并显示结果
    print(f"{'排名':<4} {'类别':<15} {'有车激活':<10} {'场景激活':<10} {'车辆特异性':<12} {'相对提升':<10} {'判断'}")
    print("-" * 85)

    # 按车辆特异性排序
    sorted_by_specificity = sorted(vehicle_specific_scores.items(), key=lambda x: x[1]["specificity"], reverse=True)

    true_vehicle_classes = []

    for rank, (target_idx, scores) in enumerate(sorted_by_specificity[:10], 1):  # 只显示前10名
        class_name = class_names.get(target_idx, f"class_{target_idx}")

        # 判断是否为真正的车辆类别
        is_vehicle = scores["specificity"] > 0.01 and scores["relative_boost"] > 0.1  # 特异性阈值  # 相对提升阈值

        if is_vehicle:
            true_vehicle_classes.append((target_idx, class_name, scores["specificity"]))
            judgment = "✓ 真实车辆"
        else:
            judgment = "✗ 场景干扰"

        print(f"{rank:<4} {class_name:<15} {scores['car_activation']:<10.4f} " f"{scores['scene_activation']:<10.4f} {scores['specificity']:<12.4f} " f"{scores['relative_boost']:<10.2f} {judgment}")

    # 最终结论
    print(f"=== 车辆分类结论 ===")
    if true_vehicle_classes:
        print("检测到的真实车辆类别:")
        for i, (idx, name, score) in enumerate(true_vehicle_classes[:3], 1):
            print(f"{i}. {name} (特异性得分: {score:.4f})")

        # 最可能的车辆类型
        best_match = true_vehicle_classes[0]
        print(f"🎯 最可能的车辆类型: {best_match[1]}")
        print(f"   置信度得分: {best_match[2]:.4f}")

        # 给出解释
        if best_match[1] in ["trailer truck", "cab"]:
            print(f"   解释: 您的BMW X1 SUV被识别为{best_match[1]},")
            print(f"        这可能是因为SUV的尺寸和轮廓与商用车辆相似")

    else:
        print("⚠️  未检测到明确的车辆特异性激活")
        print("   可能原因:")
        print("   1. 模型主要依赖场景上下文而非车辆特征")
        print("   2. 需要更精确的车辆检测方法")
        print("   3. 图像中的车辆特征不够明显")

    return vehicle_specific_scores, true_vehicle_classes


def quick_vehicle_check(cam_model, image_with_car, image_scene_only, target_indices):
    """
    快速检查是否存在车辆
    """
    print("=== 快速车辆存在性检查 ===")

    total_car_activation = 0
    total_scene_activation = 0

    for i, target_idx in enumerate(target_indices):
        mask_car, _ = cam_model(image_with_car, i, "./")
        mask_scene, _ = cam_model(image_scene_only, i, "./")

        total_car_activation += torch.sum(torch.relu(mask_car)).item()
        total_scene_activation += torch.sum(torch.relu(mask_scene)).item()

    vehicle_evidence = total_car_activation - total_scene_activation

    print(f"有车场景总激活: {total_car_activation:.4f}")
    print(f"纯场景总激活: {total_scene_activation:.4f}")
    print(f"车辆证据得分: {vehicle_evidence:.4f}")

    if vehicle_evidence > 0.1:
        print("✅ 检测到车辆存在的强证据")
        return True
    elif vehicle_evidence > 0.01:
        print("⚠️  检测到车辆存在的弱证据")
        return True
    else:
        print("❌ 未检测到明显的车辆证据")
        return False


# 使用示例
if __name__ == "__main__":
    print("简单车辆分类器已准备就绪")
    print("使用方法:")
    print("1. vehicle_scores, vehicle_classes = simple_vehicle_classification(cam_model, car_img, scene_img, target_indices)")
    print("2. has_vehicle = quick_vehicle_check(cam_model, car_img, scene_img, target_indices)")
