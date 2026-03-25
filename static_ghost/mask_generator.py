from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from static_ghost.fast_inpaint import _build_mask

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
    mask = _build_mask(width, height, regions, 0, 0, dilation)
    cv2.imwrite(output_path, mask)
    return output_path
