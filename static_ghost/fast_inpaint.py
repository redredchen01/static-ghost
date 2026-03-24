"""Fast inpainting: crop watermark region, inpaint small patch, paste back.

Instead of running IOPaint on full 1920x1080 frames, we:
1. Crop a small region around the watermark (with padding)
2. Create a mask for just that crop
3. Run IOPaint on the tiny crops
4. Paste inpainted crops back onto original frames

This is 5-10x faster than full-frame processing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

from static_ghost.detector import Region


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

    # Read first frame to get dimensions
    sample = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    fh, fw = sample.shape[:2]

    # Compute bounding box that covers ALL regions + dilation + padding
    min_x = max(0, min(r.x for r in regions) - dilation - padding)
    min_y = max(0, min(r.y for r in regions) - dilation - padding)
    max_x = min(fw, max(r.x + r.w for r in regions) + dilation + padding)
    max_y = min(fh, max(r.y + r.h for r in regions) + dilation + padding)

    crop_w = max_x - min_x
    crop_h = max_y - min_y
    print(f"Crop region: ({min_x},{min_y}) {crop_w}x{crop_h} (full frame: {fw}x{fh})")
    print(f"Speedup: ~{(fw*fh)/(crop_w*crop_h):.1f}x fewer pixels per frame")

    # Create temp dirs for crops
    import tempfile
    tmp = tempfile.mkdtemp(prefix="sg_fast_")
    crops_dir = os.path.join(tmp, "crops")
    crops_out_dir = os.path.join(tmp, "crops_out")
    os.makedirs(crops_dir)
    os.makedirs(crops_out_dir)

    # Create crop-sized mask
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for r in regions:
        rx = r.x - min_x
        ry = r.y - min_y
        # Draw rectangle with dilation
        x1 = max(0, rx - dilation)
        y1 = max(0, ry - dilation)
        x2 = min(crop_w, rx + r.w + dilation)
        y2 = min(crop_h, ry + r.h + dilation)
        mask[y1:y2, x1:x2] = 255

    # Extra dilation with kernel for smooth edges
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1, dilation * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask_path = os.path.join(tmp, "crop_mask.png")
    cv2.imwrite(mask_path, mask)

    # Step 1: Crop all frames
    print(f"Cropping {len(frame_files)} frames...")
    for fname in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        crop = frame[min_y:max_y, min_x:max_x]
        cv2.imwrite(os.path.join(crops_dir, fname), crop)

    # Step 2: Run IOPaint on crops
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

    # Step 3: Paste back
    print("Compositing inpainted regions back...")
    for fname in frame_files:
        original = cv2.imread(os.path.join(frames_dir, fname))
        inpainted_crop = cv2.imread(os.path.join(crops_out_dir, fname))
        if inpainted_crop is None:
            # Fallback: copy original
            shutil.copy(os.path.join(frames_dir, fname), os.path.join(output_dir, fname))
            continue
        original[min_y:max_y, min_x:max_x] = inpainted_crop
        cv2.imwrite(os.path.join(output_dir, fname), original)

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)
    print("Fast inpainting complete.")
