from PIL import Image
import numpy as np


def crop_edge_white_to_transparent(input_path, output_path, threshold=10):
    """
    仅去除图片外周围一圈的白色背景，保留内部内容不变，输出透明PNG
    :param input_path: 输入图片路径（jpg/png都支持）
    :param output_path: 输出PNG路径
    :param threshold: 白色判定阈值，默认适配你的新手标牌图
    """
    # 读取图片转成RGBA格式，转成numpy数组方便处理
    img = Image.open(input_path)
    img = img.convert("RGBA")
    img_arr = np.array(img)
    height, width = img_arr.shape[:2]

    # 标记哪些像素需要改成透明：从四个边缘出发，连通的白色都标记
    need_transparent = np.zeros((height, width), dtype=bool)
    from collections import deque

    q = deque()

    # 四个方向偏移量
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 把边缘所有白色像素加入队列，作为起点
    # 遍历上下两条边
    for x in range(width):
        if np.all(img_arr[0, x, :3] > threshold):
            if not need_transparent[0, x]:
                need_transparent[0, x] = True
                q.append((0, x))
        if np.all(img_arr[height - 1, x, :3] > threshold):
            if not need_transparent[height - 1, x]:
                need_transparent[height - 1, x] = True
                q.append((height - 1, x))
    # 遍历左右两条边
    for y in range(height):
        if np.all(img_arr[y, 0, :3] > threshold):
            if not need_transparent[y, 0]:
                need_transparent[y, 0] = True
                q.append((y, 0))
        if np.all(img_arr[y, width - 1, :3] > threshold):
            if not need_transparent[y, width - 1]:
                need_transparent[y, width - 1] = True
                q.append((y, width - 1))

    # 广度优先搜索：把边缘连通的所有白色都标记为透明
    while q:
        y, x = q.popleft()
        for dy, dx in dirs:
            ny = y + dy
            nx = x + dx
            # 判断坐标在图片范围内，还没标记过，且是白色
            if 0 <= ny < height and 0 <= nx < width:
                if not need_transparent[ny, nx] and np.all(img_arr[ny, nx, :3] > threshold):
                    need_transparent[ny, nx] = True
                    q.append((ny, nx))

    # 把标记好的像素改成透明（alpha通道设为0）
    img_arr[need_transparent, 3] = 0

    # 转回Image对象保存
    result_img = Image.fromarray(img_arr)
    result_img.save(output_path, "PNG")
    print(f"处理完成！仅去除边缘一圈白色，结果保存到: {output_path}")


# ------------------- 你的路径已经替换好了 -------------------
if __name__ == "__main__":
    INPUT_JPG = "/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/America.jpg"
    OUTPUT_PNG = "/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/America.png"
    # 如果你的图边缘白色偏暗，可以把threshold改成200再试
    crop_edge_white_to_transparent(INPUT_JPG, OUTPUT_PNG, threshold=220)
