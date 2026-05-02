#!/bin/bash
# 下一步增大学习率！


# echo "[$(date)] 开始等待6小时..."
# sleep 6h
# echo "[$(date)] 等待结束，启动任务。"

# 定义日志目录和参数
STYLE_IMAGE="LP.png" # 试一下shixi
CONTENT_IMAGE="LP.png"
VEHICLE="BMW.png"
PAINT_MASK="29"
STEPS=300
LEARNING_RATE=(0.25 0.3 0.35 0.4 0.45 0.25 0.25 0.25 0.25)
CONTENT_WEIGHT=(3600 3600 3600 3600 3600 3600 3600 3600)
STYLE_WEIGHT=(20000 20000 20000 20000 20000 20000 20000 20000)
ADV_TYPE="yolo"
ADV_WEIGHT=(1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000)
TV_WEIGHT=(0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003 0.0000000003)
BATCH_SIZE=6
MASK_WEIGHT=1000.0
DEPTH_MODEL="monodepth2"
REALISTIC_WEIGHT=(0.000000001 0.0000000001 0.0000000001 0.0000000001 0.000000001 0.0000000001 0.0000000001 0.0000000001)
LOG_POSTFIX="mono_car_Rob_disp"
BASELINE="proposed"
STYLE_LAMBDA=2
END_LEARNING_RATE=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
DECAY_STEPS=0.7
DECAY_POWER=(2 2 2 2 2 2 2 2) # 先快后慢
DEVICE=0
NPS_WEIGHT=0.001
FIXED_LOCATION=1
CLASS_LAMBDA=(0.0001 0.0001 0.0001 0.0001 0.0001)

LOG_DIR="/home/hyj/code/MDE_Attack/DeepPhotoStyle_pytorch/log1104/total/${CONTENT_IMAGE}_${PAINT_MASK}_${STEPS}"
LOG_FILE="$LOG_DIR/grad_${CONTENT_IMAGE}_${PAINT_MASK}_${STEPS}_${LEARNING_RATE}.txt"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

echo "start" >> "$LOG_FILE"


# 运行多个实验，输出到不同的日志文件
for i in {0,1,2,3,4}; do
    RESULT_FILE="$LOG_DIR/grad_${STYLE_IMAGE}_${CLASS_LAMBDA[$i]}_${LEARNING_RATE[$i]}_${i}.txt"

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
            -dp "${DECAY_POWER[$i]}"
            -nw "$NPS_WEIGHT"
            -fl "$FIXED_LOCATION"
            -cl "${CLASS_LAMBDA[$i]}"" >> "$LOG_FILE"
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
        -fl "$FIXED_LOCATION" \
        -cl "${CLASS_LAMBDA[$i]}" \
        > "$RESULT_FILE" 2>&1
    echo "启动实验 $i，日志输出到 $RESULT_FILE" >> "$LOG_FILE"
    wait
    python show_adv.py "$RESULT_FILE"
    echo "已可视化实验 $i 的结果" >> "$LOG_FILE"

done

echo "所有实验已启动，请检查日志目录 $LOG_DIR" >> "$LOG_FILE"