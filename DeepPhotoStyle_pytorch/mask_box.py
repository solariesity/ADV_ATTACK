from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MASK_DIR = PROJECT_ROOT / "asset" / "src_img" / "car"


def get_box_from_mask(mask_path, foreground_threshold=0):
    """
    Infer box=(left, top, right, bottom) from a mask image.

    Foreground rule:
    - Grayscale: pixel value > foreground_threshold
    - RGB: any channel > foreground_threshold
    - RGBA: any RGB channel > foreground_threshold and alpha > 0
    """
    with Image.open(mask_path) as img:
        img_array = np.array(img)

    if img_array.ndim == 2:
        foreground_mask = img_array > foreground_threshold
    elif img_array.ndim == 3 and img_array.shape[2] == 4:
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        foreground_mask = np.any(rgb > foreground_threshold, axis=2) & (alpha > 0)
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        foreground_mask = np.any(img_array > foreground_threshold, axis=2)
    else:
        raise ValueError(f"Unsupported mask shape: {img_array.shape}")

    if not np.any(foreground_mask):
        raise ValueError(f"No foreground pixels found in mask: {mask_path}")

    ys, xs = np.where(foreground_mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resolve_mask_path(mask_path, mask_dir, prefix, num, ext):
    if mask_path:
        return Path(mask_path)

    mask_dir = Path(mask_dir)
    direct_match = mask_dir / f"{prefix}{num}{ext}"
    if direct_match.exists():
        return direct_match

    pattern = re.compile(rf"^{re.escape(prefix)}0*{num}{re.escape(ext)}$")
    matches = sorted(path for path in mask_dir.glob(f"{prefix}*{ext}") if pattern.match(path.name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Cannot find mask for prefix={prefix}, num={num}, ext={ext} in {mask_dir}")
    raise FileExistsError(f"Found multiple masks for prefix={prefix}, num={num}: {matches}")


def resolve_output_key(mask_path, num):
    if num is not None:
        return str(num)

    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(mask_path).name)
    if match:
        return str(int(match.group(1)))
    return Path(mask_path).stem


def parse_args():
    parser = ArgumentParser(description="Infer box coordinates from a binary mask image.")
    parser.add_argument(
        "--mask-path",
        help="Direct path to the mask image.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=31,
        help="Mask number, used to build a file name like BMW_PaintMask31.png.",
    )
    parser.add_argument(
        "--prefix",
        default="BMW_PaintMask",
        help="Mask filename prefix used together with --num.",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(DEFAULT_MASK_DIR),
        help="Directory that stores the mask images.",
    )
    parser.add_argument(
        "--ext",
        default=".png",
        help="Mask file extension used together with --num.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Foreground threshold. Pixels above this value are treated as foreground.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mask_path = resolve_mask_path(
        mask_path=args.mask_path,
        mask_dir=args.mask_dir,
        prefix=args.prefix,
        num=args.num,
        ext=args.ext,
    )

    box = get_box_from_mask(mask_path, foreground_threshold=args.threshold)
    output_key = resolve_output_key(mask_path=mask_path, num=args.num)
    print(f"{output_key}:{box}")
