#!/usr/bin/env bash
set -euo pipefail

# Run this script after activating a Python environment that has Pillow and NumPy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${SCRIPT_DIR}/mask_box_all.txt"

cd "${SCRIPT_DIR}"

{
python mask_box.py --prefix BMW_PaintMask --num 11
python mask_box.py --prefix BMW_PaintMask --num 12
python mask_box.py --prefix BMW_PaintMask --num 13
python mask_box.py --prefix BMW_PaintMask --num 14
python mask_box.py --prefix BMW_PaintMask --num 15
python mask_box.py --prefix BMW_PaintMask --num 16
python mask_box.py --prefix BMW_PaintMask --num 17
python mask_box.py --prefix BMW_PaintMask --num 18
python mask_box.py --prefix BMW_PaintMask --num 19
python mask_box.py --prefix BMW_PaintMask --num 21
python mask_box.py --prefix BMW_PaintMask --num 22
python mask_box.py --prefix BMW_PaintMask --num 23
python mask_box.py --prefix BMW_PaintMask --num 24
python mask_box.py --prefix BMW_PaintMask --num 25
python mask_box.py --prefix BMW_PaintMask --num 26
python mask_box.py --prefix BMW_PaintMask --num 27
python mask_box.py --prefix BMW_PaintMask --num 28
python mask_box.py --prefix BMW_PaintMask --num 29
python mask_box.py --prefix BMW_PaintMask --num 30
python mask_box.py --prefix BMW_PaintMask --num 31
python mask_box.py --prefix BMW_PaintMask --num 32
python mask_box.py --prefix BMW_PaintMask --num 33
python mask_box.py --prefix BMW_PaintMask --num 34
python mask_box.py --prefix BMW_PaintMask --num 35
} > "${OUTPUT_FILE}"

printf 'Saved results to %s\n' "${OUTPUT_FILE}"
