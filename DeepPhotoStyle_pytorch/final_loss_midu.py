# 梯度安全且作用完全一样的loss_midu函数

import torch
import numpy as np
import config


def loss_midu(x1):
    """梯度安全版本的loss_midu,作用与原版完全相同"""

    x1 = torch.tanh(x1)
    cam_edge = x1.shape[0]  # 动态获取尺寸

    # 使用numpy跟踪访问状态(不影响梯度)
    vis = np.zeros((cam_edge, cam_edge), dtype=bool)

    loss = []

    for i in range(cam_edge):
        for j in range(cam_edge):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs_gradient_safe(x1, i, j, point, vis, cam_edge)

                if len(point) > 0:
                    # 关键修复1: 使用torch.stack替代reduce
                    point_tensor = torch.stack(point)
                    point_sum = point_tensor.sum()

                    # 关键修复2: 安全处理分母
                    denominator = cam_edge * cam_edge + 1 - n
                    if denominator <= 0:
                        # 如果分母为负或零,使用组件大小作为分母
                        component_loss = point_sum / max(n, 1)
                    else:
                        component_loss = point_sum / denominator

                    loss.append(component_loss)

    if len(loss) == 0:
        return torch.zeros(1).to(config.device0)

    # 关键修复3: 使用torch.stack和mean替代reduce
    if len(loss) == 1:
        return loss[0]
    else:
        loss_tensor = torch.stack(loss)
        return loss_tensor.mean()


def dfs_gradient_safe(x1, x, y, points, vis, cam_edge):
    """梯度安全的DFS函数,功能与原版完全相同"""

    points.append(x1[x][y])
    vis[x][y] = True  # 使用True而不是1,但功能相同
    n = 1

    # 四个方向的递归调用,与原版完全相同
    if x + 1 < cam_edge and x1[x + 1][y] > 0 and not vis[x + 1][y]:
        n += dfs_gradient_safe(x1, x + 1, y, points, vis, cam_edge)
    if x - 1 >= 0 and x1[x - 1][y] > 0 and not vis[x - 1][y]:
        n += dfs_gradient_safe(x1, x - 1, y, points, vis, cam_edge)
    if y + 1 < cam_edge and x1[x][y + 1] > 0 and not vis[x][y + 1]:
        n += dfs_gradient_safe(x1, x, y + 1, points, vis, cam_edge)
    if y - 1 >= 0 and x1[x][y - 1] > 0 and not vis[x][y - 1]:
        n += dfs_gradient_safe(x1, x, y - 1, points, vis, cam_edge)

    return n


# 测试函数
def test_gradient_safety():
    """测试梯度传播是否正常"""

    # 创建测试输入
    test_input = torch.randn(7, 7).requires_grad_(True)

    print("=== 测试梯度安全的loss_midu ===")
    print(f"输入形状: {test_input.shape}")
    print(f"输入范围: {test_input.min():.3f} 到 {test_input.max():.3f}")

    # 计算损失
    loss = loss_midu(test_input)
    print(f"损失值: {loss}")
    print(f"损失requires_grad: {loss.requires_grad}")
    print(f"损失grad_fn: {loss.grad_fn}")

    # 测试反向传播
    loss.backward()

    if test_input.grad is not None:
        print(f"✓ 梯度传播成功!")
        print(f"梯度范数: {test_input.grad.norm()}")
        print(f"梯度形状: {test_input.grad.shape}")
        print(f"梯度范围: {test_input.grad.min():.6f} 到 {test_input.grad.max():.6f}")
    else:
        print("✗ 梯度传播失败!")

    return loss


def test_extreme_cases():
    """测试极端情况"""

    print("=== 测试极端情况 ===")

    # 测试用例1: 全正值
    test1 = torch.ones(7, 7) * 5
    test1.requires_grad_(True)
    loss1 = loss_midu(test1)
    print(f"全正值测试: {loss1}")

    # 测试用例2: 全负值
    test2 = torch.ones(7, 7) * -5
    test2.requires_grad_(True)
    loss2 = loss_midu(test2)
    print(f"全负值测试: {loss2}")

    # 测试用例3: 混合值
    test3 = torch.randn(7, 7) * 2
    test3.requires_grad_(True)
    loss3 = loss_midu(test3)
    print(f"混合值测试: {loss3}")

    # 测试梯度传播
    for i, (name, loss) in enumerate([("全正值", loss1), ("全负值", loss2), ("混合值", loss3)]):
        try:
            loss.backward(retain_graph=True)
            print(f"{name}: ✓ 梯度传播正常")
        except Exception as e:
            print(f"{name}: ✗ 梯度传播失败 - {e}")


if __name__ == "__main__":
    test_gradient_safety()
    test_extreme_cases()
