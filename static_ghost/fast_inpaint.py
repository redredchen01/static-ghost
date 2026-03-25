"""Fast inpainting: crop watermark region, inpaint small patch, paste back.

Optimizations over naive full-frame approach:
1. Crop-and-paste: only inpaint the watermark region (~15x fewer pixels)
2. JPEG temp files: 3-5x faster I/O vs PNG, ~5x smaller files
3. Multiprocess crop/paste: parallelize CPU-bound image operations
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

from static_ghost.detector import Region

# JPEG quality for temp crops (95 = visually lossless, much faster than PNG)
_JPEG_QUALITY = 95


def _crop_one(args: tuple) -> None:
    """Crop a single frame to the watermark region. (picklable for multiprocessing)"""
    src_path, dst_path, min_x, min_y, max_x, max_y = args
    frame = cv2.imread(src_path)
    crop = frame[min_y:max_y, min_x:max_x]
    cv2.imwrite(dst_path, crop, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])


def _paste_one(args: tuple) -> None:
    """Paste inpainted crop back onto original frame. Output as PNG (for FFmpeg merge)."""
    orig_path, crop_path, dst_path, min_x, min_y, max_x, max_y = args
    original = cv2.imread(orig_path)
    inpainted_crop = cv2.imread(crop_path)
    if inpainted_crop is None:
        shutil.copy(orig_path, dst_path)
        return
    original[min_y:max_y, min_x:max_x] = inpainted_crop
    cv2.imwrite(dst_path, original)


def fast_remove(
    frames_dir: str,
    output_dir: str,
    regions: list[Region],
    dilation: int,
    device: str = "cpu",
    padding: int = 50,
) -> None:
    """Crop-inpaint-paste pipeline for fast watermark removal."""
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frame_files:
        raise ValueError(f"No PNG frames found in {frames_dir}")

    sample = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    fh, fw = sample.shape[:2]

    min_x = max(0, min(r.x for r in regions) - dilation - padding)
    min_y = max(0, min(r.y for r in regions) - dilation - padding)
    max_x = min(fw, max(r.x + r.w for r in regions) + dilation + padding)
    max_y = min(fh, max(r.y + r.h for r in regions) + dilation + padding)

    crop_w = max_x - min_x
    crop_h = max_y - min_y
    print(f"Crop region: ({min_x},{min_y}) {crop_w}x{crop_h} (full frame: {fw}x{fh})")
    print(f"Speedup: ~{(fw*fh)/(crop_w*crop_h):.1f}x fewer pixels per frame")

    tmp = tempfile.mkdtemp(prefix="sg_fast_")
    crops_dir = os.path.join(tmp, "crops")
    crops_out_dir = os.path.join(tmp, "crops_out")
    os.makedirs(crops_dir)
    os.makedirs(crops_out_dir)

    # Create crop-sized mask
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for r in regions:
        rx, ry = r.x - min_x, r.y - min_y
        x1 = max(0, rx - dilation)
        y1 = max(0, ry - dilation)
        x2 = min(crop_w, rx + r.w + dilation)
        y2 = min(crop_h, ry + r.h + dilation)
        mask[y1:y2, x1:x2] = 255

    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1, dilation * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask_path = os.path.join(tmp, "crop_mask.png")
    cv2.imwrite(mask_path, mask)

    # Step 1: Parallel crop
    nproc = min(cpu_count(), 8)
    print(f"Cropping {len(frame_files)} frames ({nproc} workers)...")
    crop_args = [
        (
            os.path.join(frames_dir, fname),
            os.path.join(crops_dir, os.path.splitext(fname)[0] + ".jpg"),
            min_x, min_y, max_x, max_y,
        )
        for fname in frame_files
    ]
    with Pool(nproc) as pool:
        pool.map(_crop_one, crop_args)

    # Step 2: IOPaint on crops
    print("Running IOPaint on cropped regions...")
    cmd = [
        "iopaint", "run",
        "--model", "lama",
        "--device", device,
        "--image", crops_dir,
        "--mask", mask_path,
        "--output", crops_out_dir,
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Parallel paste-back
    print(f"Compositing {len(frame_files)} frames ({nproc} workers)...")
    paste_args = []
    for fname in frame_files:
        crop_stem = os.path.splitext(fname)[0]
        # IOPaint output may be .jpg or .png depending on input extension
        crop_out = os.path.join(crops_out_dir, crop_stem + ".jpg")
        if not os.path.exists(crop_out):
            crop_out = os.path.join(crops_out_dir, crop_stem + ".png")
        paste_args.append((
            os.path.join(frames_dir, fname),
            crop_out,
            os.path.join(output_dir, fname),
            min_x, min_y, max_x, max_y,
        ))
    with Pool(nproc) as pool:
        pool.map(_paste_one, paste_args)

    shutil.rmtree(tmp, ignore_errors=True)
    print("Fast inpainting complete.")
