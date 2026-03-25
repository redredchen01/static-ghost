"""Fast inpainting: stream-based crop-inpaint-paste pipeline.

Instead of extracting all frames to disk:
1. FFmpeg pipe → read frames in memory → crop watermark region → save only crops (JPEG)
2. IOPaint processes the small crop images
3. FFmpeg pipe → read frames in memory → paste inpainted crop → pipe to FFmpeg encoder

This eliminates full-frame temp files (~95% less disk, ~30-50% faster).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from static_ghost.detector import Region

_JPEG_QUALITY = 95


def fast_remove(
    frames_dir: str,
    output_dir: str,
    regions: list[Region],
    dilation: int,
    device: str = "cpu",
    padding: int = 50,
) -> None:
    """Crop-inpaint-paste pipeline. Accepts frames_dir for backward compat."""
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
    crop_w, crop_h = max_x - min_x, max_y - min_y

    print(f"Crop region: ({min_x},{min_y}) {crop_w}x{crop_h} (full frame: {fw}x{fh})")
    print(f"Speedup: ~{(fw*fh)/(crop_w*crop_h):.1f}x fewer pixels per frame")

    tmp = tempfile.mkdtemp(prefix="sg_fast_")
    crops_dir = os.path.join(tmp, "crops")
    crops_out_dir = os.path.join(tmp, "crops_out")
    os.makedirs(crops_dir)
    os.makedirs(crops_out_dir)

    mask = _build_mask(crop_w, crop_h, regions, min_x, min_y, dilation)
    mask_path = os.path.join(tmp, "crop_mask.png")
    cv2.imwrite(mask_path, mask)

    feather_alpha, feather_inv = _build_feather_masks(crop_w, crop_h, feather_px=8)

    # Step 1: Crop all frames
    print(f"Cropping {len(frame_files)} frames...")
    for fname in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        crop = frame[min_y:max_y, min_x:max_x]
        crop_name = os.path.splitext(fname)[0] + ".jpg"
        cv2.imwrite(os.path.join(crops_dir, crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])

    # Step 2: IOPaint on crops
    print("Running IOPaint on cropped regions...")
    _run_iopaint(crops_dir, mask_path, crops_out_dir, device)

    # Step 3: Paste back — pre-build crop path lookup
    crop_lookup = _build_crop_lookup(crops_out_dir, frame_files)

    print(f"Compositing {len(frame_files)} frames with edge blending...")
    for fname in frame_files:
        original = cv2.imread(os.path.join(frames_dir, fname))
        crop_stem = os.path.splitext(fname)[0]
        crop_out = crop_lookup.get(crop_stem)

        if crop_out:
            inpainted = cv2.imread(crop_out)
            original_crop = original[min_y:max_y, min_x:max_x]
            original[min_y:max_y, min_x:max_x] = _feather_blend(original_crop, inpainted, feather_alpha, feather_inv)

        cv2.imwrite(os.path.join(output_dir, fname), original)

    shutil.rmtree(tmp, ignore_errors=True)
    print("Fast inpainting complete.")


def fast_remove_streamed(
    video_path: str,
    output_path: str,
    regions: list[Region],
    dilation: int,
    device: str = "cpu",
    padding: int = 50,
) -> None:
    """Stream-based pipeline: no full-frame temp files. ~95% less disk usage.

    Pass 1: FFmpeg decode → crop in memory → save only crops
    IOPaint: process crops
    Pass 2: FFmpeg decode → paste inpainted crop → FFmpeg encode
    """
    from static_ghost.video_engine import probe

    meta = probe(video_path)
    fw, fh = meta["width"], meta["height"]
    fps = meta["fps"]

    min_x = max(0, min(r.x for r in regions) - dilation - padding)
    min_y = max(0, min(r.y for r in regions) - dilation - padding)
    max_x = min(fw, max(r.x + r.w for r in regions) + dilation + padding)
    max_y = min(fh, max(r.y + r.h for r in regions) + dilation + padding)
    crop_w, crop_h = max_x - min_x, max_y - min_y

    print(f"Crop region: ({min_x},{min_y}) {crop_w}x{crop_h} (full frame: {fw}x{fh})")
    print(f"Speedup: ~{(fw*fh)/(crop_w*crop_h):.1f}x fewer pixels per frame")

    tmp = tempfile.mkdtemp(prefix="sg_stream_")
    crops_dir = os.path.join(tmp, "crops")
    crops_out_dir = os.path.join(tmp, "crops_out")
    os.makedirs(crops_dir)
    os.makedirs(crops_out_dir)

    mask = _build_mask(crop_w, crop_h, regions, min_x, min_y, dilation)
    mask_path = os.path.join(tmp, "crop_mask.png")
    cv2.imwrite(mask_path, mask)

    feather_alpha, feather_inv = _build_feather_masks(crop_w, crop_h, feather_px=8)

    # Pass 1: Decode video → crop → save crops only
    print("Pass 1: Extracting watermark crops...")
    frame_count = _extract_crops(video_path, fw, fh, crops_dir, min_x, min_y, max_x, max_y)
    print(f"Extracted {frame_count} crops.")

    # IOPaint on crops
    print("Running IOPaint on cropped regions...")
    _run_iopaint(crops_dir, mask_path, crops_out_dir, device)

    # Pre-build crop path lookup for Pass 2
    crop_lookup = {}
    for f in os.listdir(crops_out_dir):
        stem = os.path.splitext(f)[0]
        crop_lookup[stem] = os.path.join(crops_out_dir, f)

    # Pass 2: Decode video again → paste inpainted crops → encode output
    print("Pass 2: Compositing and encoding...")
    _paste_and_encode(video_path, output_path, fw, fh, fps,
                      crop_lookup, min_x, min_y, max_x, max_y,
                      feather_alpha, feather_inv)

    shutil.rmtree(tmp, ignore_errors=True)
    print("Stream inpainting complete.")


def _run_iopaint(image_dir: str, mask_path: str, output_dir: str, device: str) -> None:
    """Run IOPaint with stderr captured."""
    cmd = [
        "iopaint", "run",
        "--model", "lama",
        "--device", device,
        "--image", image_dir,
        "--mask", mask_path,
        "--output", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"IOPaint failed (exit {result.returncode}): {result.stderr[:500]}")


def _build_crop_lookup(crops_out_dir: str, frame_files: list[str]) -> dict[str, str]:
    """Pre-build stem → path mapping for inpainted crops."""
    lookup = {}
    for f in os.listdir(crops_out_dir):
        stem = os.path.splitext(f)[0]
        lookup[stem] = os.path.join(crops_out_dir, f)
    return lookup


def _extract_crops(
    video_path: str, fw: int, fh: int,
    crops_dir: str, min_x: int, min_y: int, max_x: int, max_y: int,
) -> int:
    """Decode video via FFmpeg pipe, crop watermark region, save as JPEG."""
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-threads", "0",
            "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    frame_size = fw * fh * 3
    idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((fh, fw, 3))
            crop = frame[min_y:max_y, min_x:max_x].copy()
            cv2.imwrite(
                os.path.join(crops_dir, f"frame_{idx:06d}.jpg"),
                crop, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
            )
            idx += 1
            if idx % 1000 == 0:
                print(f"  Cropped {idx} frames...", flush=True)
    finally:
        proc.stdout.close()
        proc.wait()
    return idx


def _paste_and_encode(
    video_path: str, output_path: str, fw: int, fh: int, fps: float,
    crop_lookup: dict[str, str],
    min_x: int, min_y: int, max_x: int, max_y: int,
    feather_alpha: np.ndarray, feather_inv: np.ndarray,
) -> None:
    """Decode video, paste inpainted crops, encode to output."""
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-threads", "0",
            "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    tmp_video = output_path + ".tmp_noaudio.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{fw}x{fh}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-v", "quiet",
            tmp_video,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    frame_size = fw * fh * 3
    idx = 0
    try:
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((fh, fw, 3)).copy()

            stem = f"frame_{idx:06d}"
            crop_path = crop_lookup.get(stem)

            if crop_path:
                inpainted = cv2.imread(crop_path)
                if inpainted is not None:
                    original_crop = frame[min_y:max_y, min_x:max_x]
                    frame[min_y:max_y, min_x:max_x] = _feather_blend(
                        original_crop, inpainted, feather_alpha, feather_inv
                    )

            encoder.stdin.write(frame.tobytes())
            idx += 1
            if idx % 1000 == 0:
                print(f"  Encoded {idx} frames...", flush=True)
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()

    # Mux: add original audio
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "copy",
            "-v", "quiet",
            output_path,
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_video):
        os.remove(tmp_video)


def _build_mask(
    crop_w: int, crop_h: int,
    regions: list[Region], offset_x: int, offset_y: int, dilation: int,
) -> np.ndarray:
    """Build binary mask for the crop region."""
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for r in regions:
        rx, ry = r.x - offset_x, r.y - offset_y
        x1 = max(0, rx - dilation)
        y1 = max(0, ry - dilation)
        x2 = min(crop_w, rx + r.w + dilation)
        y2 = min(crop_h, ry + r.h + dilation)
        mask[y1:y2, x1:x2] = 255
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1, dilation * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _build_feather_masks(crop_w: int, crop_h: int, feather_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Build feathered alpha + inverse masks. Pre-expanded for broadcast."""
    mask = np.ones((crop_h, crop_w), dtype=np.float32)
    for i in range(feather_px):
        alpha = (i + 1) / (feather_px + 1)
        if i < crop_h // 2:
            mask[i, :] = np.minimum(mask[i, :], alpha)
            mask[crop_h - 1 - i, :] = np.minimum(mask[crop_h - 1 - i, :], alpha)
        if i < crop_w // 2:
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, crop_w - 1 - i] = np.minimum(mask[:, crop_w - 1 - i], alpha)
    # Pre-expand to (H, W, 1) and pre-compute inverse — avoids per-frame allocation
    alpha_3d = mask[:, :, np.newaxis]
    inv_3d = (1.0 - mask)[:, :, np.newaxis]
    return alpha_3d, inv_3d


def _feather_blend(original: np.ndarray, inpainted: np.ndarray,
                   alpha: np.ndarray, alpha_inv: np.ndarray) -> np.ndarray:
    """Blend inpainted region using pre-computed alpha/inverse masks.

    Uses uint8 → float32 only where needed, with pre-computed inverse to avoid
    per-frame (1 - alpha) allocation.
    """
    blended = inpainted.astype(np.float32) * alpha + original.astype(np.float32) * alpha_inv
    return blended.astype(np.uint8)
