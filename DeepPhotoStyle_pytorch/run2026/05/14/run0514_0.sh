#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=400

LABEL=(
    "fix_cw2356_300_cw14_20"
    "fix_cw2356_300_cw14_22"
    "fix_cw2356_300_cw14_25"
    "fix_cw2356_300_cw14_27"
    "fix_cw14_20_cw2356_300"
    "fix_cw14_20_cw2356_200"
    "fix_cw14_20_cw2356_100"
    "fix_cw14_20_cw2356_50"
)

LEARNING_RATE=(0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35)
CONTENT_WEIGHT=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)
ADV_TYPE="yolo"
ADV_WEIGHT=(1 1 1 1 1 1 1 1)
TV_WEIGHT=(0 0 0 0 0 0 0 0)
BATCH_SIZE=6
DEPTH_MODEL="monodepth2"
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.014 0.014 0.014 0.014 0.014 0.014 0.014 0.014)
DECAY_STEPS=0.8
DECAY_POWER=(2 2 2 2 2 2 2 2)
NPS_WEIGHT=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001)
RANDOM_SCENE=(0 0 0 0 0 0 0 0)
COLOR_POWER=2
OFFICIAL_YOLO_TENSOR=1
MIDU_WEIGHT=(50 50 50 50 50 50 50 50)

STYLE_WEIGHT=(0 0 0 0 0 0 0 0)
MASK_WEIGHT=(0 0 0 0 0 0 0 0)
REALISTIC_WEIGHT=(0 0 0 0 0 0 0 0)
COLOR_WEIGHT=(0 0 0 0 0 0 0 0)
ORIGINAL_WEIGHT=(0 0 0 0 0 0 0 0)

DEVICE=0
# COLOR_WEIGHT_14=(20 22 25 27 20 20 20 20)
# COLOR_WEIGHT_2356=(300 300 300 300 300 200 100 50)
COLOR_WEIGHT_14=(0 0 0 0 0 0 0 0)
COLOR_WEIGHT_2356=(0 0 0 0 0 0 0 0)

LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0514/color_204_0_t/color_scan_${CONTENT_IMAGE}_${STYLE_IMAGE}_${PAINT_MASK}_0"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


for i in {0,}; do
    RESULT_FILE="$LOG_DIR/${LABEL[$i]}_LR_${LEARNING_RATE[$i]}_CP_${COLOR_POWER}_CW14_${COLOR_WEIGHT_14[$i]}_CW2356_${COLOR_WEIGHT_2356[$i]}_MIDU_WEIGHT_${MIDU_WEIGHT[$i]}_${i}.txt"

    RS_FLAG=""
    if [ "${RANDOM_SCENE[$i]}" == "1" ]; then
        RS_FLAG="--random-scene"
    fi

    echo "python /home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/test.py \
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
            -ds "$DECAY_STEPS" \
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
            -oyt "$OFFICIAL_YOLO_TENSOR"" >> "$LOG_FILE"

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
        -ds "$DECAY_STEPS" \
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
        > "$RESULT_FILE" 2>&1

    echo "启动实验 $i，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_plot.py --mode adv "$RESULT_FILE"
    python show_plot.py --mode total "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"
done

echo "0514 0 cw14/cw2356 扫描已完成，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"
