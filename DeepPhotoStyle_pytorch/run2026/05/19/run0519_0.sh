#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="HQ.png"
PAINT_MASK="21"
STEPS=10000

LABEL_PREFIX="america_hq_pm21_mix"
LEARNING_RATE=0.003
ADV_TYPE="yolo"
ADV_WEIGHT=1
TV_WEIGHT=0
BATCH_SIZE=6
DEPTH_MODEL="monodepth2"
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=0.02
DECAY_STEPS=0.8
DECAY_POWER=1.5
NPS_WEIGHT=0.0001
FIXED_LOCATION=0
CLASS_LAMBDA=0.0000000001
RANDOM_SCENE=1
COLOR_POWER=2
OFFICIAL_YOLO_TENSOR=1
MIDU_WEIGHT=50
OPTIMIZER_TYPE="adam"

STYLE_WEIGHT=0
MASK_WEIGHT=0
COLOR_WEIGHT=0
DEVICE=1
TARGET_IDX=""

CW14=5
CW2356=5
CONTENT_WEIGHT=0
ORIGINAL_WEIGHT=0
REALISTIC_SWEEP=(
    0.0005
)

CONTENT_TAG="${CONTENT_IMAGE%.*}"
EXPERIMENT_TAG="0_hq_cw14_5_cw2356_5_rw_0.0005_steps10000_lr0.003_cw_0_ow_0_midu50_adam"
LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0519/${EXPERIMENT_TAG}/${CONTENT_TAG}_pm${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"

idx=0
for realistic_weight in "${REALISTIC_SWEEP[@]}"; do
    if [ -n "$TARGET_IDX" ] && [ "$idx" -ne "$TARGET_IDX" ]; then
        idx=$((idx + 1))
        continue
    fi

    RESULT_FILE="$LOG_DIR/${idx}_${LABEL_PREFIX}_cw14_${CW14}_cw2356_${CW2356}_rw_${realistic_weight}_adam.txt"

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
        -tw "$TV_WEIGHT" \
        -bs "$BATCH_SIZE" \
        -mw "$MASK_WEIGHT" \
        -dm "$DEPTH_MODEL" \
        -rw "$realistic_weight" \
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
        -cw14 "$CW14" \
        -cw2356 "$CW2356" \
        -ow "$ORIGINAL_WEIGHT" \
        -fl "$FIXED_LOCATION" \
        -cl "$CLASS_LAMBDA" \
        -ml "$MIDU_WEIGHT" \
        -cp "$COLOR_POWER" \
        -oyt "$OFFICIAL_YOLO_TENSOR" \
        -ot "$OPTIMIZER_TYPE" \
        > "$RESULT_FILE" 2>&1

    echo "启动实验 $idx，cw14=$CW14，cw2356=$CW2356，realistic_weight=$realistic_weight，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_plot.py --mode adv "$RESULT_FILE"
    python show_plot.py --mode total "$RESULT_FILE"
    python show_plot.py --mode color34 "$RESULT_FILE"
    echo "已可视化实验 $idx 的结果" >> "$LOG_FILE"

    idx=$((idx + 1))
done
