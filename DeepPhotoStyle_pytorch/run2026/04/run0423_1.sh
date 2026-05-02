#!/bin/bash


# echo "[$(date)] 开始等待5小时..."
# sleep 5h
# echo "[$(date)] 等待结束，启动任务。"

# 定义日志目录和参数
STYLE_IMAGE="America.png" # 试一下shixi
CONTENT_IMAGE="America.png"
VEHICLE="BMW.png"
PAINT_MASK="34"
STEPS=400
LEARNING_RATE=(0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35)
CONTENT_WEIGHT=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)
STYLE_WEIGHT=(0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02)
ADV_TYPE="yolo"
ADV_WEIGHT=(1 1 1 1 1 1 1 1)
TV_WEIGHT=(0.0000000000000003 0.0000000000000003 0.0000000000000003 0.0000000000000003 0.0000000000000003 0.0000000000000003 0.0000000000000003 0.0000000000000003)
BATCH_SIZE=6
MASK_WEIGHT=0.001
DEPTH_MODEL="monodepth2"
REALISTIC_WEIGHT=(0.000000000000001 0.000000000000001 0.000000000000001 0.000000000000001 0.000000000000001 0.000000000000001 0.000000000000001 0.000000000000001)
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.014 0.014 0.014 0.014 0.014 0.014 0.014 0.014)
DECAY_STEPS=0.8
DECAY_POWER=(2 2 2 2 2 2 2 2) # 先快后慢
DEVICE=0
NPS_WEIGHT=(0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001)
COLOR_WEIGHT=(45 60 80 100 120 150 180 220)
ORIGINAL_WEIGHT=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)
FIXED_LOCATION=0
CLASS_LAMBDA=(0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001 0.0000000001)
MIDU_WEIGHT=(110 110 110 110 110 110 110 110)


LOG_DIR="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/log/log2026/04/log0423/color_1/color_only_scan_${CONTENT_IMAGE}_${STYLE_IMAGE}_${PAINT_MASK}"
LOG_FILE="$LOG_DIR/style_weight_change_${PAINT_MASK}_${STEPS}.txt"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


# 运行多个实验，输出到不同的日志文件
for i in {0,}; do
    RESULT_FILE="$LOG_DIR/COLOR_WEIGHT_${COLOR_WEIGHT[$i]}_MIDU_WEIGHT_${MIDU_WEIGHT[$i]}_${i}.txt"

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
            -mw "$MASK_WEIGHT" \
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
        -mw "$MASK_WEIGHT" \
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

echo "所有实验已启动，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"
