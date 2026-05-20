from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


NAME_MAP = {
    "侧": "side",
    "中": "middle",
    "近": "close",
    "近2": "close_2",
    "远": "far",
    "近到中": "close_to_middle",
    "近到中2": "close_to_middle_2",
    "中到近": "middle_to_close",
    "中到近2": "middle_to_close_2",
    "远到中": "far_to_middle",
    "远到近": "far_to_close",
    "近到远": "close_to_far",
}


def translate_video_stem(stem: str) -> str:
    if stem in NAME_MAP:
        return NAME_MAP[stem]
    return stem.strip().lower().replace(" ", "_")


def extract_frames(input_root: Path, output_root: Path) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    video_paths = sorted(input_root.rglob("*.mp4"))

    if not video_paths:
        print(f"No mp4 files found under: {input_root}")
        return

    for video_path in video_paths:
        parent_name = video_path.parent.name.lower()
        english_stem = translate_video_stem(video_path.stem)
        output_pattern = output_root / f"{parent_name}_{english_stem}_frame_%06d.png"

        print(f"Extracting: {video_path}")
        print(f"Output: {output_pattern}")

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                str(output_pattern),
            ],
            check=True,
        )

    print(f"Done. Frames saved to: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from vehicle-scene videos.")
    parser.add_argument(
        "--input-root",
        default="/data/hyj/advpatch/data/assets",
        help="Root directory containing input mp4 videos.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/hyj/advpatch/data/vehicle_scenes",
        help="Directory where extracted frames will be saved.",
    )
    args = parser.parse_args()

    extract_frames(Path(args.input_root), Path(args.output_root))


if __name__ == "__main__":
    main()
