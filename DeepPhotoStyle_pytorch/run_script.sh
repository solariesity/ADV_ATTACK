#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/script.sh"
    exit 1
fi

SCRIPT_PATH="$(readlink -f "$1")"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Script not found: $SCRIPT_PATH"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/log/1"
LOG_FILE="$LOG_DIR/$(basename "${SCRIPT_PATH%.sh}").txt"

mkdir -p "$LOG_DIR"
nohup "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &

echo "Started: $SCRIPT_PATH"
echo "PID: $!"
echo "Log: $LOG_FILE"
