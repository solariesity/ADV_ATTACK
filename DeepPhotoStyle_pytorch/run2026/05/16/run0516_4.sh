#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=600

LABEL_PREFIX="america_bmw_pm34_mix"
LEARNING_RATE=0.01
CONTENT_WEIGHT=0
ADV_TYPE="yolo"
ADV_WEIGHT=1
BATCH_SIZE=6
DEPTH_MODEL="monodepth2"
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=0.02
DECAY_STEPS=0.8
DECAY_POWER=1.5
NPS_WEIGHT=0.0001
FIXED_LOCATION=1
CLASS_LAMBDA=0.0000000001
RANDOM_SCENE=0
COLOR_POWER=2
OFFICIAL_YOLO_TENSOR=1
MIDU_WEIGHT=50
OPTIMIZER_TYPE="adam"

STYLE_WEIGHT=0
MASK_WEIGHT=0
REALISTIC_WEIGHT=0
COLOR_WEIGHT=0
ORIGINAL_WEIGHT=0

DEVICE=0
CW14_FIXED=5
CW2356_FIXED=5

mapfile -t TV_SWEEP < <(python - <<'PY'
import numpy as np
vals = np.geomspace(1e-7, 1, 30)
for v in vals:
    print(f"{v:.8g}")
PY
)

CONTENT_TAG="${CONTENT_IMAGE%.*}"
EXPERIMENT_TAG="4_bmw_cw14_5_cw2356_5_tvw_log30_midu50_adam"
LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0516/${EXPERIMENT_TAG}/${CONTENT_TAG}_pm${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"

for i in "${!TV_SWEEP[@]}"; do
    tv_value="${TV_SWEEP[$i]}"
    RESULT_FILE="$LOG_DIR/${i}_${LABEL_PREFIX}_cw14_${CW14_FIXED}_cw2356_${CW2356_FIXED}_tv_${tv_value}_adam.txt"

    RS_FLAG=""
    if [ "$RANDOM_SCENE" == "1" ]; then
        RS_FLAG="--random-scene"
    fi

    nohup python /home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/test.py \
        -s "$STYLE_IMAGE" \
        -c "$CONTENT_IMAGE" \
        -v "$VEHICLE" \
        -pm "$PAINT_MASK" \
        --steps "$STEPS" \
        -lr "$LEARNING_RATE" \
        -cw "$CONTENT_WEIGHT" \
        -sw "$STYLE_WEIGHT" \
        -at "$ADV_TYPE" \
        -aw "$ADV_WEIGHT" \
        -tw "$tv_value" \
        -bs "$BATCH_SIZE" \
        -mw "$MASK_WEIGHT" \
        -dm "$DEPTH_MODEL" \
        -rw "$REALISTIC_WEIGHT" \
        $RS_FLAG \
        -lp "$LOG_POSTFIX" \
        --late-start \
        -bl "$BASELINE" \
        -sl "$STYLE_LAMBDA" \
        -elr "$END_LEARNING_RATE" \
        -ds "$DECAY_STEPS" \
        -d "$DEVICE" \
        -dp "$DECAY_POWER" \
        -nw "$NPS_WEIGHT" \
        -clw "$COLOR_WEIGHT" \
        -cw14 "$CW14_FIXED" \
        -cw2356 "$CW2356_FIXED" \
        -ow "$ORIGINAL_WEIGHT" \
        -fl "$FIXED_LOCATION" \
        -cl "$CLASS_LAMBDA" \
        -ml "$MIDU_WEIGHT" \
        -cp "$COLOR_POWER" \
        -oyt "$OFFICIAL_YOLO_TENSOR" \
        -ot "$OPTIMIZER_TYPE" \
        > "$RESULT_FILE" 2>&1

    echo "启动实验 $i，tv_weight=$tv_value，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_plot.py --mode adv "$RESULT_FILE"
    python show_plot.py --mode total "$RESULT_FILE"
    python show_plot.py --mode color34 "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"
done
