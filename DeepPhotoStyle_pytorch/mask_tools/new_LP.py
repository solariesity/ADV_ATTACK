from PIL import Image

# 1. 打开原图片（请确保LP.png在当前目录下）
try:
    img = Image.open("/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/LP.png")
except FileNotFoundError:
    print("错误：未找到LP.png，请检查文件路径！")
    exit()

# 2. 获取原图片的宽高（w: 原宽，h: 原高）
w, h = img.size

# 3. 创建新图片（3倍宽，原高）
# - 模式选RGBA（支持透明，中间"空"即为透明）
# - 背景色设为(0,0,0,0)（完全透明）
new_width = 3 * w
new_height = h
new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

# 4. 粘贴原图片到新图片的左右两侧
# - 左边：起始坐标(0, 0)（左上角）
# - 右边：起始坐标(2*w, 0)（左边占w宽，中间留w宽，右边从2w开始）
new_img.paste(img, (0, 0))  # 左边粘贴原图片
new_img.paste(img, (2 * w, 0))  # 右边粘贴原图片

# 5. 保存新图片（命名为new_lp.png，可自行修改）
new_img.save("/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/content/new_LP.png")

print("新图片生成成功！路径：new_lp.png")
