# MDE_Attack

This repository contains a customized experimental pipeline for generating physical adversarial patches against monocular depth estimation related perception tasks. The current runnable entry point is `DeepPhotoStyle_pytorch/test.py`, which combines:

- masked style transfer on the patch region,
- vehicle or object composition into road scenes,
- adversarial optimization driven by YOLO-based feedback,
- optional random-scene and random-placement training,
- TensorBoard logging and exported visualization results.

The codebase is no longer identical to the original ECCV 2022 release. This README describes the project as it is currently organized in this workspace.

![overview](overview.png)

## Upstream basis and local modifications

This project is based on prior public research code for the ECCV 2022 paper "Physical Attack on Monocular Depth Estimation in Autonomous Driving with Optimal Adversarial Patches", and has been further modified for local experiments.

Compared with the original reference implementation, this workspace includes custom changes such as:

- using `DeepPhotoStyle_pytorch/test.py` as the main experimental entry,
- additional YOLO-driven optimization and evaluation logic,
- extra color-related masks and losses,
- revised logging, experiment scripts, and helper utilities,
- local dataset, path, and asset organization changes.

If you reuse this repository, please make it clear that this is a derived and modified codebase rather than an untouched copy of the original release.

## Project structure

```text
MDE_Attack/
├─ DeepPhotoStyle_pytorch/
│  ├─ test.py                  # current main entry
│  ├─ model.py                 # optimization loop and loss composition
│  ├─ image_preprocess.py      # image, mask, and scene preprocessing
│  ├─ my_utils.py              # paths, mask helpers, image save/load utilities
│  ├─ attach_cars.py           # paste object into scene with fixed/random placement
│  ├─ dataLoader.py            # KITTI-based scene loader for random-scene training
│  ├─ my_yolov5/               # YOLOv5 wrappers for tensor/image inference
│  ├─ asset/
│  │  ├─ src_img/
│  │  │  ├─ style/
│  │  │  ├─ content/
│  │  │  ├─ car/
│  │  │  └─ scene/
│  │  └─ gen_img/              # auto-generated resized/cropped assets
│  ├─ log/                     # experiment outputs
│  ├─ output_test/             # auxiliary test outputs
│  └─ run2025/, run2026/       # saved command history / experiment scripts
├─ models/                     # pretrained weights used by the pipeline
├─ networks/                   # depth encoder/decoder definitions
├─ pseudo_lidar/               # extra assets and validation-related files
└─ README.md
```

## What the main pipeline does

Running `DeepPhotoStyle_pytorch/test.py` performs the following steps:

1. Parse CLI arguments for style/content/object images, mask mode, optimization hyperparameters, and logging options.
2. Load and preprocess:
   - a style image and `*_StyleMask`,
   - a content image and multiple semantic masks,
   - an object image and its paint mask,
   - a road scene image used as background.
3. Build masked style/content losses with VGG19 features.
4. Optimize the patch texture with adversarial, style, content, TV, realism, NPS, color, original-image, and CAM-based losses.
5. Paste the optimized patch back onto the object, then paste the object into a scene.
6. Run YOLOv5 on intermediate/final results and save visualizations.
7. Write logs, images, and the executed command into the experiment output folder.

## Entry point

Use:

```bash
cd DeepPhotoStyle_pytorch
python test.py -h
```

Important: `test.py` should be run from inside `DeepPhotoStyle_pytorch/`.  
The preprocessing code uses `os.getcwd()` to resolve `asset/src_img/...` and `asset/gen_img/...`, so running it from the repository root will point to the wrong directories.

## Environment

This repository currently contains multiple environment files from different stages of the project:

- `env.yml`
- `requirements.txt`
- `install.txt`

They are not fully consistent with each other. For the current code, treat `install.txt` as the closer reference for the actively edited setup, and adjust based on your CUDA / PyTorch environment.

Typical runtime dependencies used by the current pipeline include:

- Python
- PyTorch
- torchvision
- torchaudio
- OpenCV
- NumPy
- SciPy
- matplotlib
- Pillow
- tensorboard
- tensorboardX
- tqdm
- neural_renderer

If you are building a fresh environment, verify imports by trying:

```bash
cd DeepPhotoStyle_pytorch
python test.py -h
```

If this command fails, install the missing packages one by one according to the traceback.

## Required path configuration

Before training, update the hard-coded paths in [DeepPhotoStyle_pytorch/my_utils.py](<D:/lab/mde_attack/代码备份/2026/0502/MDE_Attack/DeepPhotoStyle_pytorch/my_utils.py:12>):

```python
kitti_object_path = "/path/to/KITTI/object/"
project_root = "/path/to/MDE_Attack/"
log_dir = "/path/to/output/log/root"
```

These variables are used by:

- random-scene training via `dataLoader.py`,
- validation scene lookup inside `model.py`,
- image/log export paths.

## Input asset conventions

All source assets are expected under:

```text
DeepPhotoStyle_pytorch/asset/src_img/
├─ style/
├─ content/
├─ car/
└─ scene/
```

Generated resized/cropped copies are written automatically into:

```text
DeepPhotoStyle_pytorch/asset/gen_img/
```

### 1. Style images

Put the following files in `asset/src_img/style/`:

- `NAME.png`
- `NAME_StyleMask.png`

Example:

- `Warnning.png`
- `Warnning_StyleMask.png`

### 2. Content images

Put the following files in `asset/src_img/content/`:

- `NAME.png`
- `NAME_ContentMask.png`

The current `test.py` pipeline also tries to load color-specific masks:

- `NAME_WhiteMask.png`
- `NAME_RedMask.png`
- `NAME_BlackMask.png`
- `NAME_YellowMask.png`

If any mask file is missing, preprocessing falls back to an all-one mask for that file.

### 3. Object images

Put the following files in `asset/src_img/car/`:

- `NAME.png`
- `NAME_CarMask.png`
- optional fixed patch masks such as `NAME_PaintMask11.png`, `NAME_PaintMask12.png`, etc.

Examples already present in the project include:

- `BMW.png`
- `Pedestrain2.png`
- `TrafficBarrier2.png`

### 4. Scene images

Put background road scenes in `asset/src_img/scene/`.

The current preprocessing crops scenes to:

- width: `1024`
- height: `320`

The default scene used by `test.py` is `VW01.png`.

## Model weights

The current code expects weights in the repository-level `models/` directory:

- `models/mono+stereo_1024x320/`
- `models/yolov5s.pt`
- cached YOLO wrappers such as `models/yolov5s_1_0.pth` and `models/yolov5s_2_0.0.pth`

### Monodepth / depth model

`DeepPhotoStyle_pytorch/depth_model.py` loads Monodepth2-style weights from:

```text
models/mono+stereo_1024x320/
├─ encoder.pth
├─ depth.pth
├─ pose.pth
└─ pose_encoder.pth
```

### YOLOv5

`DeepPhotoStyle_pytorch/my_yolov5/yolov5_model.py` uses:

- `models/yolov5s.pt` as the base local YOLOv5 checkpoint,
- then caches deserialized variants into `models/yolov5s_1_<device>.pth` and `models/yolov5s_2_<device>.pth`.

If the cached files do not exist, the code attempts to construct them automatically.

## KITTI dataset for random-scene training

If you use `--random-scene`, the training loop reads scene images from `kitti_object_path` using `DeepPhotoStyle_pytorch/dataLoader.py`.

Expected structure:

```text
KITTI/object/
├─ training/
│  ├─ image_2/
│  └─ label_2/
├─ vehicle_detection/
│  ├─ training.txt
│  └─ testing.txt
├─ train.txt
├─ val.txt
└─ test.txt
```

The loader reads `vehicle_detection/training.txt` and `vehicle_detection/testing.txt` by default.

## Main command example

Example command adapted to the current `test.py` interface:

```bash
cd DeepPhotoStyle_pytorch

python test.py \
  -s new_LP.png \
  -c new_LP.png \
  -v BMW.png \
  -pm 33 \
  --steps 300 \
  -lr 0.45 \
  -cw 30000 \
  -sw 20000 \
  -at yolo \
  -aw 1000000 \
  -tw 0.0000000003 \
  -bs 6 \
  -mw 1000 \
  -dm monodepth2 \
  -rw 0.000000001 \
  --random-scene \
  -lp mono_car_Rob_disp \
  --late-start \
  -bl proposed \
  -sl 2 \
  -elr 0.01 \
  -ds 0.7 \
  -d 1 \
  -dp 2 \
  -nw 0.001 \
  -clw 500000 \
  -ow 10000000000 \
  -fl 1 \
  -cl 0.0001 \
  -ml 500000000
```

This example is based on the saved command history already present in `DeepPhotoStyle_pytorch/run2026/command.txt`.

## Frequently used arguments

Core inputs:

- `-s`, `--style_image`: style image filename
- `-c`, `--content_image`: content image filename
- `-v`, `--vehicle`: object image filename
- `-pm`, `--paint-mask`: mask mode or fixed paint mask id

Optimization:

- `--steps`
- `-lr`, `--learning-rate`
- `-sw`, `--style-weight`
- `-cw`, `--content-weight`
- `-aw`, `--adv-weight`
- `-rw`, `--rl-weight`
- `-tw`, `--tv-weight`
- `-nw`, `--nps-weight`
- `-clw`, `--color-weight`
- `-ow`, `--original-weight`
- `-ml`, `--midu-weight`

Training behavior:

- `--random-scene`
- `--late-start`
- `-bl`, `--baseline`
- `-dm`, `--depth-model`
- `-at`, `--adv-type`
- `-fl`, `--fixed-location`

Learning rate schedule:

- `-elr`, `--end-learning-rate`
- `-ds`, `--decay-steps`
- `-dp`, `--decay-power`

Device selection:

- `--gpu`
- `-d`, `--device`

## Paint mask modes

The code in `my_utils.py` currently supports several paint-mask modes:

- positive ids like `11`, `12`, `33`: load a fixed file such as `BMW_PaintMask33.png`
- `-1`: optimize a soft mask initialized from a half-mask
- `-2`: optimize rectangular borders
- `-3`: optimize multiple grid rectangles
- `-4`: optimize a square region based on object category

## Outputs

Each run creates a timestamped experiment directory under:

```text
<log_dir>/logs/<timestamp><log_postfix>/
```

Typical outputs include:

- `command.txt`
- TensorBoard event files
- `adv_scene_output.png`
- `car_scene_output.png`
- `adv_car_output.png`
- `yolo_result.jpg`
- `eval_adv_scene_disp.jpg`
- `adv_scene_result.jpg`

The script also logs images and scalars through `tensorboardX.SummaryWriter`.

To inspect training:

```bash
tensorboard --logdir /path/to/log_dir/logs
```

## Related scripts

- `DeepPhotoStyle_pytorch/test.py`: current primary training/evaluation entry
- `DeepPhotoStyle_pytorch/my_main.py`: older main script with a similar but not identical interface
- `DeepPhotoStyle_pytorch/test_car.py`: test path focused on car composition
- `DeepPhotoStyle_pytorch/test_color.py`: color-related experiments
- `DeepPhotoStyle_pytorch/run_mask_box_all.sh`: helper script to summarize multiple paint mask boxes

## Known caveats

- Several paths are hard-coded and must be changed before running on another machine.
- The repository contains historical environment files that do not fully match each other.
- Some comments and script names still reflect older experiment stages.
- A few modules still contain Linux-style absolute paths in example code blocks.
- The current code relies on running from `DeepPhotoStyle_pytorch/`, not from the repository root.

## Citation

If you use this project, please cite the original paper that this repository is based on, and mention that your code or experiments were built on a modified derivative of the original implementation.

```bibtex
@article{cheng2022physical,
  doi = {10.48550/ARXIV.2207.04718},
  url = {https://arxiv.org/abs/2207.04718},
  author = {Cheng, Zhiyuan and Liang, James and Choi, Hongjun and Tao, Guanhong and Cao, Zhiwen and Liu, Dongfang and Zhang, Xiangyu},
  title = {Physical Attack on Monocular Depth Estimation in Autonomous Driving with Optimal Adversarial Patches},
  publisher = {European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```
