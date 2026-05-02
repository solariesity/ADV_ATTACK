#!/bin/bash


# echo "[$(date)] 开始等待3小时..."
# sleep 3h
# echo "[$(date)] 等待结束，启动任务。"

# 定义日志目录和参数
STYLE_IMAGE="Australia.png" # 试一下shixi
CONTENT_IMAGE="Australia.png"
VEHICLE="BMW.png"
PAINT_MASK="32"
STEPS=300
LEARNING_RATE=(0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45)
CONTENT_WEIGHT=(10000 10000 10000 10000 10000 10000 10000 10000)
STYLE_WEIGHT=(20000 20000 20000 20000 20000 20000 20000 20000)
ADV_TYPE="yolo"
ADV_WEIGHT=(1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000)
TV_WEIGHT=(0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003)
BATCH_SIZE=6
MASK_WEIGHT=1000.0
DEPTH_MODEL="monodepth2"
REALISTIC_WEIGHT=(0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001 0.000000001)
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
DECAY_STEPS=0.7
DECAY_POWER=(2 2 2 2 2 2 2 2) # 先快后慢
DEVICE=1
NPS_WEIGHT=0.001
COLOR_WEIGHT=(10000000 150000000 20000000 25000000 30000000 35000000 40000000 45000000)
ORIGINAL_WEIGHT=(1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000)
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
MIDU_WEIGHT=(500000000 500000000 500000000 500000000 500000000 500000000 500000000 500000000)


LOG_DIR="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/log2026/03/log0320/color_2/midu_weight_${MIDU_WEIGHT[0]}_${CONTENT_IMAGE}_${STYLE_IMAGE}_${PAINT_MASK}"
LOG_FILE="$LOG_DIR/style_weight_change_${PAINT_MASK}_${STEPS}.txt"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


# 运行多个实验，输出到不同的日志文件
for i in {0,1,2,3,4,5,6,7}; do
    RESULT_FILE="$LOG_DIR/COLOR_WEIGHT_${COLOR_WEIGHT[$i]}_${i}.txt"

    echo "python /home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/test2.py \
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
            -nw "$NPS_WEIGHT" \
            -clw "${COLOR_WEIGHT[$i]}" \
            -ow "${ORIGINAL_WEIGHT[$i]}" \
            -fl "$FIXED_LOCATION" \
            -cl "${CLASS_LAMBDA[$i]}" \
            -ml "${MIDU_WEIGHT[$i]}"" >> "$LOG_FILE"
    nohup python /home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/test2.py \
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
        -nw "$NPS_WEIGHT" \
        -clw "${COLOR_WEIGHT[$i]}" \
        -ow "${ORIGINAL_WEIGHT[$i]}" \
        -fl "$FIXED_LOCATION" \
        -cl "${CLASS_LAMBDA[$i]}" \
        -ml "${MIDU_WEIGHT[$i]}" \
        > "$RESULT_FILE" 2>&1
    echo "启动实验 $i，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_content.py "$RESULT_FILE"
    python show_adv.py "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"

done

echo "所有实验已启动，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"