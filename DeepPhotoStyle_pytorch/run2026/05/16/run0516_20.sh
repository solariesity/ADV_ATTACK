#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=600

LABEL_PREFIX="america_bmw_pm34_mix"
LEARNING_RATE=0.01
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
FIXED_LOCATION=1
CLASS_LAMBDA=0.0000000001
RANDOM_SCENE=0
COLOR_POWER=2
OFFICIAL_YOLO_TENSOR=1
MIDU_WEIGHT=50
OPTIMIZER_TYPE="adam"

STYLE_WEIGHT=0
MASK_WEIGHT=0
COLOR_WEIGHT=0
DEVICE=1

CW14=5
CW2356=5
CONTENT_WEIGHT=0
ORIGINAL_WEIGHT=0
REALISTIC_SWEEP=(
    100
    31.62277660
    10
    3.16227766
    1
    0.31622777
    0.1
    0.03162278
    0.01
    0.00316228
    0.001
    0.00031623
    0.0001
    0.00003162
    0.00001
)

CONTENT_TAG="${CONTENT_IMAGE%.*}"
EXPERIMENT_TAG="20_bmw_cw14_5_cw2356_5_rw_100_div_sqrt10_to_1e-5_cw_0_ow_0_midu50_adam"
LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0516/${EXPERIMENT_TAG}/${CONTENT_TAG}_pm${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"

idx=0
for realistic_weight in "${REALISTIC_SWEEP[@]}"; do
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
