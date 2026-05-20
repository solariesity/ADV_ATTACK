# Loss 忍受阈值记录

## 当前完整参数口径

### 通用参数

- `style_image = America.png`
- `content_image = America.png`
- `steps = 600`
- `learning_rate = 0.01`
- `end_learning_rate = 0.02`
- `decay_steps = 0.8`
- `decay_power = 1.5`
- `optimizer = adam`
- `batch_size = 6`
- `device = 1`
- `adv_type = yolo`
- `adv_weight = 1`
- `depth_model = monodepth2`
- `baseline = proposed`
- `log_postfix = mono_car_Rob_disp`
- `official_yolo_tensor = 1`
- `late_start = true`
- `style_weight = 0`
- `mask_weight = 0`
- `tv_weight = 0`
- `color_weight / clw = 0`
- `content_weight / cw = 0`
- `original_weight / ow = 0`
- `cw14 = 5`
- `cw2356 = 5`
- `midu_weight / ml = 50`
- `nps_weight = 0.0001`
- `class_lambda = 1e-10`
- `style_lambda = 2`
- `fixed_location = 1`
- `random_scene = 1`
- `color_power = 2`

### 当前记录

| vehicle | paint_mask | rw | 备注 |
|------|------|------|------|
| `BMW.png` | `34` | `0.0022` | 当前这条线的主要参考口径 |
| `HQ.png` | `21` | `0.0005` | 新增 |
