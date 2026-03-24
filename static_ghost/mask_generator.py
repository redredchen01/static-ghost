from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from static_ghost.detector import Region


def create_mask(
    width: int,
    height: int,
    regions: list[Region],
    output_path: str,
    dilation: int = 5,
) -> str:
    """Generate a black/white mask PNG. White = watermark regions (dilated)."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for r in regions:
        draw.rectangle([r.x, r.y, r.x + r.w - 1, r.y + r.h - 1], fill=255)

    if dilation > 0:
        arr = np.array(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1, dilation * 2 + 1))
        arr = cv2.dilate(arr, kernel, iterations=1)
        mask = Image.fromarray(arr)

    mask.save(output_path)
    return output_path
