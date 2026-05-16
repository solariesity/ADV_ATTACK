#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=600

LABEL_PREFIX="america_bmw_pm34_mix"
LABEL_PARAM=(
    "cw14_0_cw2356_0_adam"
)

LEARNING_RATE=(0.005)
CONTENT_WEIGHT=(0)
ADV_TYPE="yolo"
ADV_WEIGHT=(1)
TV_WEIGHT=(0)
BATCH_SIZE=6
DEPTH_MODEL="monodepth2"
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.02)
DECAY_STEPS=(0.8)
DECAY_POWER=(1.5)
NPS_WEIGHT=(0.0001)
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0000000001)
RANDOM_SCENE=(0)
COLOR_POWER=2
OFFICIAL_YOLO_TENSOR=1
MIDU_WEIGHT=(50)
OPTIMIZER_TYPE="adam"

STYLE_WEIGHT=(0)
MASK_WEIGHT=(0)
REALISTIC_WEIGHT=(0)
COLOR_WEIGHT=(0)
ORIGINAL_WEIGHT=(0)

DEVICE=1
COLOR_WEIGHT_14=(0)
COLOR_WEIGHT_2356=(0)

CONTENT_TAG="${CONTENT_IMAGE%.*}"
EXPERIMENT_TAG="11_bmw_cw14_0_cw2356_0_midu50_adam"
LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0515/${EXPERIMENT_TAG}/${CONTENT_TAG}_pm${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"

for i in 0; do
    RESULT_FILE="$LOG_DIR/${i}_${LABEL_PREFIX}_${LABEL_PARAM[$i]}.txt"

    RS_FLAG=""
    if [ "${RANDOM_SCENE[$i]}" == "1" ]; then
        RS_FLAG="--random-scene"
    fi

    nohup python /home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/test.py \
        -s "$STYLE_IMAGE" \
        -c "$CONTENT_IMAGE" \
        -v "$VEHICLE" \
        -pm "$PAINT_MASK" \
        --steps "$STEPS" \
        -lr "${LEARNING_RATE[$i]}" \
        -cw "${CONTENT_WEIGHT[$i]}" \
        -sw "${STYLE_WEIGHT[$i]}" \
        -at "$ADV_TYPE" \
        -aw "${ADV_WEIGHT[$i]}" \
        -tw "${TV_WEIGHT[$i]}" \
        -bs "$BATCH_SIZE" \
        -mw "${MASK_WEIGHT[$i]}" \
        -dm "$DEPTH_MODEL" \
        -rw "${REALISTIC_WEIGHT[$i]}" \
        $RS_FLAG \
        -lp "$LOG_POSTFIX" \
        --late-start \
        -bl "$BASELINE" \
        -sl "$STYLE_LAMBDA" \
        -elr "${END_LEARNING_RATE[$i]}" \
        -ds "${DECAY_STEPS[$i]}" \
        -d "$DEVICE" \
        -dp "${DECAY_POWER[$i]}" \
        -nw "${NPS_WEIGHT[$i]}" \
        -clw "${COLOR_WEIGHT[$i]}" \
        -cw14 "${COLOR_WEIGHT_14[$i]}" \
        -cw2356 "${COLOR_WEIGHT_2356[$i]}" \
        -ow "${ORIGINAL_WEIGHT[$i]}" \
        -fl "$FIXED_LOCATION" \
        -cl "${CLASS_LAMBDA[$i]}" \
        -ml "${MIDU_WEIGHT[$i]}" \
        -cp "$COLOR_POWER" \
        -oyt "$OFFICIAL_YOLO_TENSOR" \
        -ot "$OPTIMIZER_TYPE" \
        > "$RESULT_FILE" 2>&1

    echo "启动实验 $i，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_plot.py --mode adv "$RESULT_FILE"
    python show_plot.py --mode total "$RESULT_FILE"
    python show_plot.py --mode color34 "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"
done
