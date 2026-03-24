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
) -> list[Region]:
    """Detect static regions via multi-frame differencing.

    Returns regions sorted by area (largest first).
    """
    frames_gray = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sample_paths]
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

    global_static = (static_counts == total_pairs).astype(np.uint8) * 255

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
        confidence = 1.0
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
