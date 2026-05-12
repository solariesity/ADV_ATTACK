#!/bin/bash


SCRIPT_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/run2026/05"
LOG_DIR="/home/hyj/code/ADV_ATTACK/DeepPhotoStyle_pytorch/log/log2026/05/log0513"
LOG_FILE="$LOG_DIR/run0513_204_chain.txt"

mkdir -p "$LOG_DIR"

echo "start run0513_204 chain" >> "$LOG_FILE"
echo "run0513_204_0.sh begin" >> "$LOG_FILE"

bash "$SCRIPT_DIR/run0513_204_0.sh"
STATUS_0=$?

echo "run0513_204_0.sh end with status $STATUS_0" >> "$LOG_FILE"

if [ "$STATUS_0" -ne 0 ]; then
    echo "run0513_204_0.sh failed, skip run0513_204_1.sh" >> "$LOG_FILE"
    exit "$STATUS_0"
fi

echo "run0513_204_1.sh begin" >> "$LOG_FILE"

bash "$SCRIPT_DIR/run0513_204_1.sh"
STATUS_1=$?

echo "run0513_204_1.sh end with status $STATUS_1" >> "$LOG_FILE"

exit "$STATUS_1"
