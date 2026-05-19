# Loss 忍受阈值记录

> 版本说明：本文件按当前这套实验重新记录，不再直接沿用更早那批参数扫描的结论。
> 适用范围：`America_pm34`、`color_power=2`、当前 `run0516/run0517` 这一批实验口径。

## 使用原则

- 这里只记录“当前版本下的经验判断”，不是绝对数学阈值。
- 如果 loss 定义、颜色集、mask、`cw14/cw2356`、`content/origin` 约束发生明显变化，需要重新记一版。
- 判断时不要只看单个 loss，仍要结合 `adv_loss`、最终可视化效果和收敛稳定性一起看。

## 当前推荐参数背景

- `cw14 = 5`
- `cw2356 = 5`
- `realistic_weight (rl/rw) = 0.0022` 左右可以作为当前一版的优先起点
- `midu_weight = 50`

备注：

- 上面这组不是“全局最优”，而是当前这批实验里较适合作为阈值参考的参数背景。
- 如果后续把 `rw` 明显改大或改小，下面这些 `color_loss3/color_loss4` 的经验阈值也要结合新的可视化结果重新理解。

## 当前完整参数口径

### 输入与任务

- `style_image = America.png`
- `content_image = America.png`
- `vehicle = BMW.png`
- `paint_mask = 34`
- `label_prefix = america_bmw_pm34_mix`

### 优化与训练

- `steps = 600`
- `learning_rate = 0.01`
- `end_learning_rate = 0.02`
- `decay_steps = 0.8`
- `decay_power = 1.5`
- `optimizer = adam`
- `batch_size = 6`
- `device = 1`
- `target_idx = 1`

### 攻击与模型

- `adv_type = yolo`
- `adv_weight = 1`
- `depth_model = monodepth2`
- `baseline = proposed`
- `log_postfix = mono_car_Rob_disp`
- `official_yolo_tensor = 1`
- `late_start = true`

### 各项权重

- `style_weight = 0`
- `mask_weight = 0`
- `tv_weight = 0`
- `color_weight / clw = 0`
- `content_weight / cw = 0`
- `original_weight / ow = 0`
- `realistic_weight / rw = 0.0022` 左右优先关注
- `cw14 = 5`
- `cw2356 = 5`
- `midu_weight / ml = 50`
- `nps_weight = 0.0001`
- `class_lambda = 1e-10`
- `style_lambda = 2`

### 其它开关

- `fixed_location = 1`
- `random_scene = 0`
- `color_power = 2`

### 当前 rw 细扫脚本范围

来自 `run0517_1.sh`：

- `rw sweep = [0.00268270, 0.00227673, 0.00193260, 0.00164049, 0.00139252, 0.00118216]`
- 当前补跑设置：`target_idx = 1`

说明：

- 这里的“完整参数口径”主要对应当前 `run0517_1.sh` 这条线。
- 如果后面把 `device`、`target_idx` 或 sweep 列表改掉，阈值本身未必失效，但“参考背景”会发生变化。

## 当前重点指标

### color_loss3

| 范围 | 判断 |
|------|------|
| `< 0.0005` | 比较理想 |
| `0.0005 ~ 0.0008` | 可接受 |
| `0.0008 ~ 0.0010` | 稍微偏大一点 |
| `> 0.0010` | 明显偏大，需要警惕 |

### color_loss4

| 范围 | 判断 |
|------|------|
| `< 0.0005` | 比较理想 |
| `0.0005 ~ 0.0008` | 可接受 |
| `0.0008 ~ 0.0010` | 稍微偏大一点 |
| `> 0.0010` | 明显偏大，需要警惕 |

## 当前参考点

参考日志：

- `log/log2026/05/log0516/5_bmw_cw14_cw2356_log30_tvw0_midu50_adam/America_pm34/9_america_bmw_pm34_mix_cw14_10.811559_cw2356_10.811559_tv_0_adam.txt`

对应记录：

- `color_loss3 = 0.00086`
- `color_loss4 = 0.00085`

当前判断：

- 这两个值都不是离谱的大，但已经可以记为“稍微偏大一点”。
- 如果后续能往 `0.0008` 以下压，会更稳妥。
- 如果继续升到 `0.0010` 以上，就要优先怀疑颜色约束开始偏松，或者视觉效果已经在往不理想方向走。

## 和当前参数的关系

- 在 `cw14=5`、`cw2356=5`、`midu=50` 这条线上，`rw` 已经显示出比较明显的影响。
- 按目前观察，`rw` 在 `0.0022` 附近更像是“还能保持一定攻击性，同时开始施加真实感约束”的过渡区域。
- 也就是说，这个文档里的阈值可以优先拿来判断 `rw≈0.0022` 一带的结果是否偏松、偏紧，参考意义会更强。

## 暂时结论

- 当前这一版实验里，`color_loss3` 和 `color_loss4` 最好都尽量压在 `0.0008` 以下。
- `0.00086 / 0.00085` 可以作为“边缘偏大”的提醒值。
- 后续如果找到攻击效果和视觉效果都更好的点，可以再用新的参考值覆盖这里。
