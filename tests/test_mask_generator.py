import cv2
import numpy as np
from pathlib import Path
from static_ghost.detector import Region
from static_ghost.mask_generator import create_mask


def test_create_mask_basic(tmp_path):
    regions = [Region(x=10, y=20, w=50, h=30, confidence=1.0)]
    output = create_mask(320, 240, regions, str(tmp_path / "mask.png"), dilation=0)
    assert Path(output).exists()
    mask = cv2.imread(output, cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (240, 320)
    assert mask[20, 10] == 255
    assert mask[49, 59] == 255
    assert mask[0, 0] == 0
    assert mask[239, 319] == 0


def test_create_mask_with_dilation(tmp_path):
    regions = [Region(x=100, y=100, w=20, h=20, confidence=1.0)]
    output = create_mask(320, 240, regions, str(tmp_path / "mask.png"), dilation=5)
    mask = cv2.imread(output, cv2.IMREAD_GRAYSCALE)
    assert mask[95, 95] == 255  # 5px outside original region
    assert mask[0, 0] == 0


def test_create_mask_multiple_regions(tmp_path):
    regions = [
        Region(x=10, y=10, w=20, h=20, confidence=1.0),
        Region(x=200, y=150, w=30, h=30, confidence=0.8),
    ]
    output = create_mask(320, 240, regions, str(tmp_path / "mask.png"), dilation=0)
    mask = cv2.imread(output, cv2.IMREAD_GRAYSCALE)
    assert mask[15, 15] == 255
    assert mask[160, 210] == 255
    assert mask[100, 100] == 0
