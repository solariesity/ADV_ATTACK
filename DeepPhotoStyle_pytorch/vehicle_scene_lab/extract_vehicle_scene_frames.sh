#!/usr/bin/env bash

set -euo pipefail

INPUT_ROOT="${1:-/data/hyj/advpatch/data/assets}"
OUTPUT_ROOT="${2:-/data/hyj/advpatch/data/vehicle_scenes}"

translate_video_name() {
    local base_name="$1"
    case "$base_name" in
        "侧") echo "side" ;;
        "中") echo "middle" ;;
        "近") echo "close" ;;
        "近2") echo "close_2" ;;
        "远") echo "far" ;;
        "近到中") echo "close_to_middle" ;;
        "近到中2") echo "close_to_middle_2" ;;
        "中到近") echo "middle_to_close" ;;
        "中到近2") echo "middle_to_close_2" ;;
        "远到中") echo "far_to_middle" ;;
        "远到近") echo "far_to_close" ;;
        "近到远") echo "close_to_far" ;;
        *)
            echo "$base_name" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g'
            ;;
    esac
}

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is not installed or not in PATH." >&2
    exit 1
fi

if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: input directory does not exist: $INPUT_ROOT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

find "$INPUT_ROOT" -type f -name "*.mp4" | sort | while read -r video_path; do
    parent_dir="$(basename "$(dirname "$video_path")")"
    video_file="$(basename "$video_path")"
    video_stem="${video_file%.mp4}"
    english_stem="$(translate_video_name "$video_stem")"
    output_prefix="${parent_dir}_${english_stem}"
    output_pattern="${OUTPUT_ROOT}/${output_prefix}_frame_%06d.png"

    echo "Extracting frames from: $video_path"
    echo "Output pattern: $output_pattern"

    ffmpeg -hide_banner -loglevel error -y -i "$video_path" "$output_pattern"
done

echo "Done. Frames are saved under: $OUTPUT_ROOT"
