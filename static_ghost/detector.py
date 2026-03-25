from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


class Region(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    confidence: float


def detect_static_regions(
    sample_paths: list[str],
    threshold: int = 15,
    *,
    num_pairs: int = 50,
    min_area: int = 100,
    max_area_ratio: float = 0.3,
    quorum: float = 0.7,
) -> list[Region]:
    """Detect static regions via multi-frame differencing.

    A pixel is considered "static" if it stays below `threshold` difference
    in at least `quorum` fraction of frame pairs (e.g., 0.7 = 70%).
    This tolerates scene changes and lighting variations in real videos.

    Returns regions sorted by area (largest first).
    """
    frames_gray = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sample_paths]
    if not frames_gray or frames_gray[0] is None:
        return []
    h, w = frames_gray[0].shape

    rng = np.random.default_rng(0)
    n = len(frames_gray)
    num_pairs = min(num_pairs, n * (n - 1) // 2)

    all_pairs = set()
    while len(all_pairs) < num_pairs:
        i, j = sorted(rng.choice(n, 2, replace=False))
        all_pairs.add((i, j))

    static_counts = np.zeros((h, w), dtype=np.int32)
    total_pairs = len(all_pairs)

    for i, j in all_pairs:
        diff = cv2.absdiff(frames_gray[i], frames_gray[j])
        static_mask = (diff < threshold).astype(np.uint8)
        static_counts += static_mask

    # Majority vote: pixel is static if it passed in >= quorum of pairs
    min_votes = int(total_pairs * quorum)
    global_static = (static_counts >= min_votes).astype(np.uint8) * 255

    # Morphological cleanup: close small gaps, remove noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    global_static = cv2.morphologyEx(global_static, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    global_static = cv2.morphologyEx(global_static, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(global_static, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = h * w
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area / total_area > max_area_ratio:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Confidence = fraction of pairs that agreed this region is static
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [cnt], -1, 255, -1)
        roi_pixels = roi_mask > 0
        avg_votes = static_counts[roi_pixels].mean() if roi_pixels.any() else 0
        confidence = round(avg_votes / total_pairs, 2)
        regions.append(Region(x=bx, y=by, w=bw, h=bh, confidence=confidence))

    regions.sort(key=lambda r: r.w * r.h, reverse=True)
    return regions


def save_preview(
    frame_path: str,
    regions: list[Region],
    output_path: str,
) -> str:
    """Draw red bounding boxes + labels on frame, save as preview PNG."""
    frame = cv2.imread(frame_path)
    for i, r in enumerate(regions):
        cv2.rectangle(frame, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 2)
        label = f"#{i+1} ({r.x},{r.y},{r.w}x{r.h}) conf={r.confidence:.2f}"
        cv2.putText(frame, label, (r.x, r.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(output_path, frame)
    return output_path
