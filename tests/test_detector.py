import cv2
import numpy as np
import pytest
from pathlib import Path
from static_ghost.detector import Region, detect_static_regions, save_preview


@pytest.fixture
def synthetic_frames(tmp_path: Path) -> list[str]:
    """Generate 10 frames: random noise backgrounds with a fixed white 40x40 block at (50,50)."""
    paths = []
    rng = np.random.default_rng(42)
    for i in range(10):
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        # Static watermark: white block at (50,50) size 40x40
        frame[50:90, 50:90] = 255
        path = str(tmp_path / f"frame_{i:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def test_detect_finds_static_watermark(synthetic_frames):
    regions = detect_static_regions(synthetic_frames, threshold=15)
    assert len(regions) >= 1
    r = regions[0]
    assert isinstance(r, Region)
    assert r.x < 60 and r.y < 60
    assert r.w > 20 and r.h > 20
    assert 0 < r.confidence <= 1.0


def test_detect_no_watermark_returns_empty(tmp_path):
    """All frames identical → entire frame is 'static' → filtered out by large-area rule."""
    paths = []
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    for i in range(10):
        path = str(tmp_path / f"frame_{i:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    regions = detect_static_regions(paths)
    assert len(regions) == 0


def test_save_preview_creates_image(synthetic_frames, tmp_path):
    regions = [Region(x=50, y=50, w=40, h=40, confidence=0.95)]
    output = save_preview(synthetic_frames[0], regions, str(tmp_path / "preview.png"))
    assert Path(output).exists()
    img = cv2.imread(output)
    assert img is not None
    assert img.shape[:2] == (240, 320)
