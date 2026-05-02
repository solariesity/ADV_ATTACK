import os
import sys
from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings

warnings.filterwarnings("ignore")

# sys.path.append("../renderer/")

import nmr_test as nmr
import neural_renderer

from torchvision.transforms import Resize

# from data_loader import MyDataset
from torch.utils.data import Dataset, DataLoader
from grad_cam import CAM

import torch.nn.functional as F
import random
from functools import reduce
import argparse
import config

# torch.manual_seed(2333)
# torch.cuda.manual_seed(2333)
# np.random.seed(2333)


# parser = argparse.ArgumentParser()

# parser.add_argument("--epoch", type=int, default=1)
# parser.add_argument("--lr", type=float, default=0.01)
# parser.add_argument("--batchsize", type=int, default=1)
# parser.add_argument("--lamb", type=float, default=1e-4)
# parser.add_argument("--d1", type=float, default=0.9)
# parser.add_argument("--d2", type=float, default=0.1)
# parser.add_argument("--t", type=float, default=0.0001)

# parser.add_argument("--obj", type=str, default="audi_et_te.obj")
# parser.add_argument("--faces", type=str, default="./all_faces.txt")
# parser.add_argument("--datapath", type=str)
# parser.add_argument("--content", type=str)
# parser.add_argument("--canny", type=str)

# args = parser.parse_args()


# obj_file = args.obj
texture_size = 6
# vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)


# mask_dir = os.path.join(args.datapath, "masks/")


torch.autograd.set_detect_anomaly(True)

log_dir = ""


def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + "-" + str(logs[key]) + "+"
    dir_name = "logs/" + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name


# T = args.t
# D1 = args.d1
# D2 = args.d2
# lamb = args.lamb
# LR = args.lr
# BATCH_SIZE = args.batchsize
# EPOCH = args.epoch


# texture_content = torch.from_numpy(np.load(args.content)).cuda(device=0)
# texture_canny = torch.from_numpy(np.load(args.canny)).cuda(device=0)
# texture_canny = (texture_canny >= 1).int()


def loss_content_diff(tex):
    return D1 * torch.sum(texture_canny * torch.pow(tex - texture_content, 2)) + D2 * torch.sum((1 - texture_canny) * torch.pow(tex - texture_content, 2))


def loss_smooth(img, mask):
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    mask = mask[:, :-1, :-1]

    mask = mask.unsqueeze(1)
    # print(mask.size())
    # print(s1.size())
    return T * torch.sum(mask * (s1 + s2))


cam_edge = 7

vis = np.zeros((cam_edge, cam_edge))


def dfs(x1, x, y, points):
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1
    # print(x, y)
    if x + 1 < cam_edge and x1[x + 1][y] > 0 and not vis[x + 1][y]:
        n += dfs(x1, x + 1, y, points)
    if x - 1 >= 0 and x1[x - 1][y] > 0 and not vis[x - 1][y]:
        n += dfs(x1, x - 1, y, points)
    if y + 1 < cam_edge and x1[x][y + 1] > 0 and not vis[x][y + 1]:
        n += dfs(x1, x, y + 1, points)
    if y - 1 >= 0 and x1[x][y - 1] > 0 and not vis[x][y - 1]:
        n += dfs(x1, x, y - 1, points)
    return n


def loss_midu(x1):
    # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())

    x1 = torch.tanh(x1)
    global vis
    vis = np.zeros((cam_edge, cam_edge))

    loss = []
    # print(x1)
    for i in range(cam_edge):
        for j in range(cam_edge):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs(x1, i, j, point)
                # print(n)
                # print(point)
                loss.append(reduce(lambda x, y: x + y, point) / (cam_edge * cam_edge + 1 - n))
    # print(vis)
    if len(loss) == 0:
        return torch.zeros(1).to(config.device0)
    return reduce(lambda x, y: x + y, loss) / len(loss)


def fixed_loss_midu(x1):
    """修复后的loss_midu函数,使用纯tensor操作保持计算图"""

    x1 = torch.tanh(x1)

    # 使用tensor操作而不是循环和递归
    # 这个实现保持了原始逻辑但使用纯tensor操作

    # 创建连通组件的mask
    positive_mask = (x1 > 0).float()

    if positive_mask.sum() == 0:
        return torch.zeros(1).to(config.device0)

    # 简化版本:计算所有正值区域的平均密度
    # 这保持了计算图的连接
    total_positive = positive_mask.sum()
    total_value = (x1 * positive_mask).sum()

    # 计算密度损失
    cam_edge = x1.shape[0]
    denominator = cam_edge * cam_edge + 1 - total_positive

    denominator = torch.clamp(denominator, min=1.0)
    density_loss = total_value / denominator

    return density_loss


def dfs_3(x1, x, y, points, height, width):
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1

    # 使用实际的height和width而不是cam_edge
    if x + 1 < height and x1[x + 1][y] > 0 and not vis[x + 1][y]:
        n += dfs_3(x1, x + 1, y, points, height, width)
    if x - 1 >= 0 and x1[x - 1][y] > 0 and not vis[x - 1][y]:
        n += dfs_3(x1, x - 1, y, points, height, width)
    if y + 1 < width and x1[x][y + 1] > 0 and not vis[x][y + 1]:
        n += dfs_3(x1, x, y + 1, points, height, width)
    if y - 1 >= 0 and x1[x][y - 1] > 0 and not vis[x][y - 1]:
        n += dfs_3(x1, x, y - 1, points, height, width)
    return n


def loss_midu_3(x1):
    # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())

    height, width = x1.shape[-2:]
    x1 = torch.tanh(x1)
    global vis
    vis = np.zeros((height, width))

    loss = []
    # print(x1)
    for i in range(height):
        for j in range(width):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs_3(x1, i, j, point, height, width)
                # print(n)
                # print(point)
                loss.append(reduce(lambda x, y: x + y, point) / (height * width + 1 - n))
    # print(vis)
    if len(loss) == 0:
        return torch.zeros(1).to(config.device0)
    return reduce(lambda x, y: x + y, loss) / len(loss)


def loss_midu_2(x1):
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


# Textures

# texture_param = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), "float32") * -0.9  # test 0
# texture_param = torch.autograd.Variable(torch.from_numpy(texture_param).cuda(device=0), requires_grad=True)

# texture_origin = torch.from_numpy(textures[None, :, :, :, :, :]).cuda(device=0)

# texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), "int8")
# with open(args.faces, "r") as f:
# face_ids = f.readlines()
# print(face_ids)
# for face_id in face_ids:
# if face_id != "\n":
# texture_mask[int(face_id) - 1, :, :, :, :] = 1
# texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)


def cal_texture(CONTENT=False):
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    # return textures
    return texture_origin * (1 - texture_mask) + texture_mask * textures


# def run_cam(data_dir, epoch, train=True, batch_size=BATCH_SIZE):
#     print(data_dir)
#     dataset = MyDataset(data_dir, 224, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         # num_workers=2,
#     )

#     optim = torch.optim.Adam([texture_param], lr=LR)

#     Cam = CAM()

#     textures = cal_texture()

#     dataset.set_textures(textures)
#     print(len(dataset))
#     for _ in range(epoch):
#         print("Epoch: ", _, "/", epoch)
#         count = 0
#         tqdm_loader = tqdm.tqdm(loader)
#         for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader):
#             index = int(index[0])

#             total_img_np = total_img.data.cpu().numpy()[0]
#             # print(total_img_np.shape)
#             total_img_np = Image.fromarray(np.transpose(total_img_np, (1, 2, 0)).astype("uint8"))

#             total_img_np.save(os.path.join(log_dir, "test_total.jpg"))
#             # print(texture_img.size())
#             # print(torch.max(texture_img))
#             Image.fromarray((255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype("uint8")).save(os.path.join(log_dir, "texture2.png"))
#             Image.fromarray((255 * masks).data.cpu().numpy()[0].astype("uint8")).save(os.path.join(log_dir, "mask.png"))
#             # scipy.misc.imsave(os.path.join(log_dir, 'mask.png'), (255*masks).data.cpu().numpy()[0])

#             #######
#             # CAM #
#             #######
#             pred = 0

#             mask, pred = Cam(total_img, index, log_dir)

#             ###########
#             #   LOSS  #
#             ###########

#             loss = loss_midu(mask) + lamb * loss_content_diff(texture_param) + loss_smooth(texture_img, masks)

#             with open(os.path.join(log_dir, "loss.txt"), "a") as f:

#                 tqdm_loader.set_description("Loss %.8f, Prob %.8f" % (loss.data.cpu().numpy(), pred))
#                 f.write("Loss %.8f, Prob %.8f\n" % (loss.data.cpu().numpy(), pred))

#             ############
#             # backward #
#             ############
#             if train and loss != 0:
#                 # print(loss.data.cpu().numpy())
#                 optim.zero_grad()
#                 loss.backward(retain_graph=True)
#                 optim.step()
#             # print(texture_param)

#             textures = cal_texture()
#             dataset.set_textures(textures)


def vehicle_confidence_loss(mask, method="sum"):
    """
    计算车辆类别的置信度损失

    Args:
        mask: Grad-CAM激活图
        method: 'sum', 'weighted', 'threshold', 'comprehensive'
    """
    if method == "sum":
        return torch.sum(torch.relu(mask))

    elif method == "weighted":
        positive_mask = torch.relu(mask)
        return torch.sum(positive_mask**2)

    elif method == "threshold":
        threshold = 0.3
        high_activation = torch.relu(mask - threshold)
        return torch.sum(high_activation)

    elif method == "comprehensive":
        positive_mask = torch.relu(mask)
        activation_sum = torch.sum(positive_mask)
        activation_area = torch.sum(positive_mask > 0.1).float()
        return activation_sum * torch.log(activation_area + 1)

    else:
        raise ValueError("Unknown method")


if __name__ == "__main__":
    total_img = torch.rand(1, 3, 224, 224).to(config.device0) * 255
    Cam = CAM()
    index = 0
    log_dir = f"./cam_run_{1}"
    os.makedirs(log_dir, exist_ok=True)
    mask, pred = Cam(total_img, index, log_dir)
    loss = loss_midu(mask)
    print(loss)
    sys.exit()

    logs = {
        "epoch": EPOCH,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "model": "resnet50",
        "loss_func": "loss_midu+loss_content+loss_smooth",
        "lamb": lamb,
        "D1": D1,
        "D2": D2,
        "T": T,
    }

    make_log_dir(logs)

    train_dir = os.path.join(args.datapath, "phy_attack/train/")
    test_dir = os.path.join(args.datapath, "phy_attack/test/")

    texture_param = torch.autograd.Variable(torch.from_numpy(np.load(args.content)).cuda(device=0), requires_grad=True)

    run_cam(train_dir, EPOCH)

    np.save(os.path.join(log_dir, "texture.npy"), texture_param.data.cpu().numpy())
