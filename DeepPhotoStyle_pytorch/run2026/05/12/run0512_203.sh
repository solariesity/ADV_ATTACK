#!/bin/bash


STYLE_IMAGE="America.png"
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=20

LABEL=(
    "cp2_cw14_40_cw2356_60000"
)

STYLE_WEIGHT=(0)
MASK_WEIGHT=(0)
REALISTIC_WEIGHT=(0)
COLOR_WEIGHT=(0.01)

LEARNING_RATE=(0.35)
CONTENT_WEIGHT=(0.01)
ADV_TYPE="yolo"
ADV_WEIGHT=(1)
TV_WEIGHT=(0)
BATCH_SIZE=6
DEPTH_MODEL="monodepth2"
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.014)
DECAY_STEPS=0.8
DECAY_POWER=(2)
DEVICE=1
NPS_WEIGHT=(0.0001)
COLOR_WEIGHT_14=(40)
COLOR_WEIGHT_2356=(60000)
ORIGINAL_WEIGHT=(0)
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0000000001)
MIDU_WEIGHT=(50)
RANDOM_SCENE=(0)
COLOR_POWER=2


LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0512/color_203/color_scan_${CONTENT_IMAGE}_${STYLE_IMAGE}_${PAINT_MASK}"
LOG_FILE="$LOG_DIR/color_scan_${PAINT_MASK}_${STEPS}.txt"

mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


for i in 0; do
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
            -cp "$COLOR_POWER"" >> "$LOG_FILE"

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
        > "$RESULT_FILE" 2>&1

    echo "启动实验 $i，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_plot.py --mode content "$RESULT_FILE"
    python show_plot.py --mode adv "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"
done

echo "cp=2 单轮实验已完成，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"
