#!/bin/bash


STYLE_IMAGE="America.png" # 试一下shixi
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=400

LABEL=(
    "color_scan_24_repeat_1"
    "color_scan_24_repeat_2"
    "color_scan_28_repeat_1"
    "color_scan_28_repeat_2"
    "color_scan_32_repeat_1"
    "color_scan_32_repeat_2"
    "color_scan_35_repeat_1"
    "color_scan_35_repeat_2"
)

LEARNING_RATE=(0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35)
CONTENT_WEIGHT=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)
STYLE_WEIGHT=(0 0 0 0 0 0 0 0)
ADV_TYPE="yolo"
ADV_WEIGHT=(1 1 1 1 1 1 1 1)
TV_WEIGHT=(0 0 0 0 0 0 0 0)
BATCH_SIZE=6
MASK_WEIGHT=(0 0 0 0 0 0 0 0)
DEPTH_MODEL="monodepth2"
REALISTIC_WEIGHT=(0 0 0 0 0 0 0 0)
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.014 0.014 0.014 0.014 0.014 0.014 0.014 0.014)
DECAY_STEPS=0.8
DECAY_POWER=(2 2 2 2 2 2 2 2)
DEVICE=1
NPS_WEIGHT=(0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001)
COLOR_WEIGHT=(24 24 28 28 32 32 35 35)
ORIGINAL_WEIGHT=(0 0 0 0 0 0 0 0)
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001)
MIDU_WEIGHT=(85 85 85 85 85 85 85 85)


LOG_DIR="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/log/log2026/04/log0430/color_2/color_scan_${CONTENT_IMAGE}_${STYLE_IMAGE}_${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


for i in {0,1,2,3,4,5,6,7}; do
    RESULT_FILE="$LOG_DIR/${LABEL[$i]}_LR_${LEARNING_RATE[$i]}_COLOR_WEIGHT_${COLOR_WEIGHT[$i]}_MIDU_WEIGHT_${MIDU_WEIGHT[$i]}_${i}.txt"

    echo "python /home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/test.py \
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
            -ow "${ORIGINAL_WEIGHT[$i]}" \
            -fl "$FIXED_LOCATION" \
            -cl "${CLASS_LAMBDA[$i]}" \
            -ml "${MIDU_WEIGHT[$i]}"" >> "$LOG_FILE"

    nohup python /home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/test.py \
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

echo "color_weight 扫描已完成，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"
