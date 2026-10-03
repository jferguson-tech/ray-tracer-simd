#!/usr/bin/env python3
"""
Compare two renders, or two folders of renders, and report how much they differ.

    python compare_images.py output_before output_after
    python compare_images.py a.png b.png --diff diff.png
    python compare_images.py output_before output_after --min-psnr 40

For each pair it prints the PSNR (higher is closer; identical images are "inf"),
the mean and the largest per-channel difference on a 0-255 scale. With
--min-psnr the exit code is 1 if any pair is below the threshold, so it can be
used as a regression check.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def load(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read {path}")
    return img.astype(np.float64)


def compare(path_a: str, path_b: str, diff_path: str = None) -> float:
    a, b = load(path_a), load(path_b)
    if a.shape != b.shape:
        raise SystemExit(f"Sizes differ: {path_a} is {a.shape[1]}x{a.shape[0]}, "
                         f"{path_b} is {b.shape[1]}x{b.shape[0]}")
    diff = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    name = os.path.basename(path_a)
    print(f"{name:28s} PSNR {psnr:6.2f} dB   mean diff {diff.mean():6.3f}   max diff {diff.max():5.0f}")
    if diff_path:
        # Differences amplified 8x so small changes are visible
        cv2.imwrite(diff_path, np.clip(diff * 8.0, 0, 255).astype(np.uint8))
    return psnr


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two renders or two folders of renders")
    ap.add_argument("a", help="image or folder")
    ap.add_argument("b", help="image or folder")
    ap.add_argument("--diff", help="write an amplified difference image (single pair only)")
    ap.add_argument("--min-psnr", type=float, help="exit with code 1 if any pair is below this PSNR")
    args = ap.parse_args()

    if os.path.isdir(args.a) != os.path.isdir(args.b):
        raise SystemExit("Give two images or two folders")
    if os.path.isdir(args.a):
        names = sorted(n for n in os.listdir(args.a)
                       if n.lower().endswith(".png") and os.path.exists(os.path.join(args.b, n)))
        if not names:
            raise SystemExit("No PNG files with the same name in both folders")
        pairs = [(os.path.join(args.a, n), os.path.join(args.b, n)) for n in names]
    else:
        pairs = [(args.a, args.b)]

    worst = float("inf")
    for pa, pb in pairs:
        worst = min(worst, compare(pa, pb, args.diff if len(pairs) == 1 else None))
    if args.min_psnr is not None and worst < args.min_psnr:
        print(f"FAIL: lowest PSNR {worst:.2f} dB is below {args.min_psnr:.2f} dB")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
