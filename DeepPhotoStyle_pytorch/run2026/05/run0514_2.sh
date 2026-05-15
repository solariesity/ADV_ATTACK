#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=600

LABEL_PREFIX="america_bmw_pm34"
LABEL_PARAM="lr0.1_midu0_2"

LEARNING_RATE=0.1
CONTENT_WEIGHT=0
ADV_TYPE="yolo"
ADV_WEIGHT=1
TV_WEIGHT=0
BATCH_SIZE=6
DEPTH_MODEL="monodepth2"
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=0.014
DECAY_STEPS=0.8
DECAY_POWER=2
NPS_WEIGHT=0.0001
FIXED_LOCATION=1
CLASS_LAMBDA=0.0000000001
RANDOM_SCENE=0
COLOR_POWER=2
OFFICIAL_YOLO_TENSOR=1
MIDU_WEIGHT=0

STYLE_WEIGHT=0
MASK_WEIGHT=0
REALISTIC_WEIGHT=0
COLOR_WEIGHT=0
ORIGINAL_WEIGHT=0

DEVICE=0
COLOR_WEIGHT_14=0
COLOR_WEIGHT_2356=0

CONTENT_TAG="${CONTENT_IMAGE%.*}"
EXPERIMENT_TAG="bmw_lr_scan"
LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0514/${EXPERIMENT_TAG}/${CONTENT_TAG}_pm${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"
RESULT_FILE="$LOG_DIR/2_${LABEL_PREFIX}_${LABEL_PARAM}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"

RS_FLAG=""
if [ "$RANDOM_SCENE" == "1" ]; then
    RS_FLAG="--random-scene"
fi

echo "python /home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/test.py \
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
        -cw14 "$COLOR_WEIGHT_14" \
        -cw2356 "$COLOR_WEIGHT_2356" \
        -ow "$ORIGINAL_WEIGHT" \
        -fl "$FIXED_LOCATION" \
        -cl "$CLASS_LAMBDA" \
        -ml "$MIDU_WEIGHT" \
        -cp "$COLOR_POWER" \
        -oyt "$OFFICIAL_YOLO_TENSOR"" >> "$LOG_FILE"

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
    -cw14 "$COLOR_WEIGHT_14" \
    -cw2356 "$COLOR_WEIGHT_2356" \
    -ow "$ORIGINAL_WEIGHT" \
    -fl "$FIXED_LOCATION" \
    -cl "$CLASS_LAMBDA" \
    -ml "$MIDU_WEIGHT" \
    -cp "$COLOR_POWER" \
    -oyt "$OFFICIAL_YOLO_TENSOR" \
    > "$RESULT_FILE" 2>&1

echo "启动实验 2，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
wait
python show_plot.py --mode adv "$RESULT_FILE"
python show_plot.py --mode total "$RESULT_FILE"
echo "已可视化实验 2 的结果" >> "$LOG_FILE"

