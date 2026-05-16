#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=400

LABEL=(
    "cw14_1.9_repeat_1"
    "cw14_1.9_repeat_2"
    "cw14_2.0_repeat_1"
    "cw14_2.0_repeat_2"
    "cw14_2.1_repeat_1"
    "cw14_2.1_repeat_2"
    "cw14_2.2_repeat_1"
    "cw14_2.2_repeat_2"
)

STYLE_WEIGHT=(0 0 0 0 0 0 0 0)
MASK_WEIGHT=(0 0 0 0 0 0 0 0)
REALISTIC_WEIGHT=(0 0 0 0 0 0 0 0)
COLOR_WEIGHT=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)

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
DEVICE=1
NPS_WEIGHT=(0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001)
COLOR_WEIGHT_14=(1.9 1.9 2.0 2.0 2.1 2.1 2.2 2.2)
COLOR_WEIGHT_2356=(18 18 18 18 18 18 18 18)
ORIGINAL_WEIGHT=(0 0 0 0 0 0 0 0)
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001)
MIDU_WEIGHT=(50 50 50 50 50 50 50 50)


LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0508/color_204_2/color_scan_${CONTENT_IMAGE}_${STYLE_IMAGE}_${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


for i in {0,1,2,3,4,5,6,7}; do
    RESULT_FILE="$LOG_DIR/${LABEL[$i]}_LR_${LEARNING_RATE[$i]}_CW14_${COLOR_WEIGHT_14[$i]}_CW2356_${COLOR_WEIGHT_2356[$i]}_MIDU_WEIGHT_${MIDU_WEIGHT[$i]}_${i}.txt"

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
            --random-scene \
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
            -ml "${MIDU_WEIGHT[$i]}"" >> "$LOG_FILE"

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
        --random-scene \
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
        > "$RESULT_FILE" 2>&1

    echo "启动实验 $i，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_plot.py --mode content "$RESULT_FILE"
    python show_plot.py --mode adv "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"
done

echo "color_weight_14 / color_weight_2356 扫描已完成，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"
