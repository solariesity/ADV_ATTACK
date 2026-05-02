# 保持原始意义但梯度安全的loss_midu函数

import torch
import numpy as np
from functools import reduce
import config

# 需要定义cam_edge
cam_edge = 7  # 根据你的实际CAM尺寸调整


def gradient_safe_dfs(x1, x, y, visited_mask, device):
    """
    使用tensor操作实现DFS,保持计算图连接
    返回连通组件的所有值和像素数量
    """
    # 使用栈而不是递归,避免深度问题
    stack = [(x, y)]
    component_values = []
    component_size = 0

    while stack:
        cx, cy = stack.pop()

        # 检查边界和访问状态
        if cx < 0 or cx >= cam_edge or cy < 0 or cy >= cam_edge or visited_mask[cx, cy] or x1[cx, cy] <= 0:
            continue

        # 标记为已访问
        visited_mask[cx, cy] = True

        # 收集值
        component_values.append(x1[cx, cy])
        component_size += 1

        # 添加4连通的邻居
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cam_edge and 0 <= ny < cam_edge and not visited_mask[nx, ny] and x1[nx, ny] > 0:
                stack.append((nx, ny))

    return component_values, component_size


def loss_midu_gradient_safe(x1):
    """
    保持原始算法意义的梯度安全版本
    """
    x1 = torch.tanh(x1)
    device = x1.device

    # 使用numpy数组跟踪访问状态(不影响梯度)
    visited = np.zeros((cam_edge, cam_edge), dtype=bool)

    loss_components = []

    # 遍历所有像素
    for i in range(cam_edge):
        for j in range(cam_edge):
            # 如果是正值且未访问过
            if x1[i, j] > 0 and not visited[i, j]:
                # 使用梯度安全的DFS
                component_values, component_size = gradient_safe_dfs(x1, i, j, visited, device)

                if len(component_values) > 0:
                    # 使用torch.stack而不是reduce来保持梯度
                    component_tensor = torch.stack(component_values)
                    component_sum = component_tensor.sum()

                    # 计算该连通组件的密度损失
                    denominator = cam_edge * cam_edge + 1 - component_size

                    # 安全处理分母
                    if denominator <= 0:
                        # 如果分母为负或零,使用组件大小作为分母
                        component_loss = component_sum / component_size
                    else:
                        component_loss = component_sum / denominator

                    loss_components.append(component_loss)

    # 如果没有找到任何连通组件
    if len(loss_components) == 0:
        return torch.zeros(1).to(device)

    # 使用torch.stack和mean而不是reduce
    if len(loss_components) == 1:
        return loss_components[0]
    else:
        loss_tensor = torch.stack(loss_components)
        return loss_tensor.mean()


def loss_midu_fixed_minimal(x1):
    """
    最小修改版本:只修复reduce调用,其他保持不变
    """
    x1 = torch.tanh(x1)
    global vis
    vis = np.zeros((cam_edge, cam_edge))

    loss = []

    for i in range(cam_edge):
        for j in range(cam_edge):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs(x1, i, j, point)

                if len(point) > 0:
                    # 修复1: 使用torch.stack而不是reduce
                    point_tensor = torch.stack(point)
                    point_sum = point_tensor.sum()

                    # 修复2: 安全处理分母
                    denominator = cam_edge * cam_edge + 1 - n
                    if denominator <= 0:
                        # 使用组件大小作为分母
                        component_loss = point_sum / n
                    else:
                        component_loss = point_sum / denominator

                    loss.append(component_loss)

    if len(loss) == 0:
        return torch.zeros(1).to(config.device0)

    # 修复3: 使用torch.stack和mean而不是reduce
    if len(loss) == 1:
        return loss[0]
    else:
        loss_tensor = torch.stack(loss)
        return loss_tensor.mean()


# 你需要的dfs函数(如果使用最小修改版本)
def dfs(x1, x, y, points):
    """原始的dfs函数"""
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1

    if x + 1 < cam_edge and x1[x + 1][y] > 0 and not vis[x + 1][y]:
        n += dfs(x1, x + 1, y, points)
    if x - 1 >= 0 and x1[x - 1][y] > 0 and not vis[x - 1][y]:
        n += dfs(x1, x - 1, y, points)
    if y + 1 < cam_edge and x1[x][y + 1] > 0 and not vis[x][y + 1]:
        n += dfs(x1, x, y + 1, points)
    if y - 1 >= 0 and x1[x][y - 1] > 0 and not vis[x][y - 1]:
        n += dfs(x1, x, y - 1, points)

    return n


def test_both_versions():
    """测试两个版本的梯度传播"""
    test_input = torch.randn(cam_edge, cam_edge).requires_grad_(True)

    print("=== 测试完全重写版本 ===")
    loss1 = loss_midu_gradient_safe(test_input)
    print(f"Loss: {loss1}")
    print(f"requires_grad: {loss1.requires_grad}")

    loss1.backward(retain_graph=True)
    if test_input.grad is not None:
        print(f"✓ 梯度传播成功: {test_input.grad.norm()}")
        test_input.grad.zero_()
    else:
        print("✗ 梯度传播失败")

    print("=== 测试最小修改版本 ===")
    loss2 = loss_midu_fixed_minimal(test_input)
    print(f"Loss: {loss2}")
    print(f"requires_grad: {loss2.requires_grad}")

    loss2.backward()
    if test_input.grad is not None:
        print(f"✓ 梯度传播成功: {test_input.grad.norm()}")
    else:
        print("✗ 梯度传播失败")


if __name__ == "__main__":
    test_both_versions()
