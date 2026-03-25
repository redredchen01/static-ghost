import cv2
import numpy as np
import pytest
from pathlib import Path
from static_ghost.detector import Region
from static_ghost.fast_inpaint import (
    _build_mask,
    _build_feather_masks,
    _feather_blend,
    _build_crop_lookup,
)


class TestBuildMask:
    def test_basic_single_region(self):
        mask = _build_mask(100, 80, [Region(10, 20, 30, 25, 1.0)], 0, 0, 0)
        assert mask.shape == (80, 100)
        assert mask[25, 15] == 255  # inside region
        assert mask[0, 0] == 0     # outside

    def test_with_offset(self):
        mask = _build_mask(50, 50, [Region(60, 70, 20, 20, 1.0)], 50, 60, 0)
        # Region at (60,70) with offset (50,60) → relative (10,10)
        assert mask[15, 15] == 255
        assert mask[0, 0] == 0

    def test_dilation_expands(self):
        mask_no_dil = _build_mask(100, 100, [Region(40, 40, 10, 10, 1.0)], 0, 0, 0)
        mask_dil = _build_mask(100, 100, [Region(40, 40, 10, 10, 1.0)], 0, 0, 5)
        assert mask_dil.sum() > mask_no_dil.sum()

    def test_multiple_regions(self):
        regions = [Region(5, 5, 10, 10, 1.0), Region(80, 60, 15, 15, 0.9)]
        mask = _build_mask(100, 80, regions, 0, 0, 0)
        assert mask[10, 10] == 255
        assert mask[65, 85] == 255
        assert mask[40, 50] == 0


class TestFeatherMasks:
    def test_shape_and_range(self):
        alpha, inv = _build_feather_masks(50, 40, feather_px=5)
        assert alpha.shape == (40, 50, 1)
        assert inv.shape == (40, 50, 1)
        assert alpha.min() > 0
        assert alpha.max() <= 1.0

    def test_edges_less_than_center(self):
        alpha, _ = _build_feather_masks(100, 80, feather_px=10)
        center_val = alpha[40, 50, 0]
        edge_val = alpha[0, 50, 0]
        assert edge_val < center_val

    def test_alpha_plus_inv_equals_one(self):
        alpha, inv = _build_feather_masks(60, 40, feather_px=8)
        total = alpha + inv
        np.testing.assert_allclose(total, 1.0)


class TestFeatherBlend:
    def test_blend_produces_valid_output(self):
        h, w = 30, 40
        alpha, inv = _build_feather_masks(w, h, feather_px=5)
        original = np.full((h, w, 3), 100, dtype=np.uint8)
        inpainted = np.full((h, w, 3), 200, dtype=np.uint8)
        result = _feather_blend(original, inpainted, alpha, inv)
        assert result.dtype == np.uint8
        assert result.shape == (h, w, 3)
        # Center should be closer to inpainted (200)
        assert result[h // 2, w // 2, 0] > 150

    def test_blend_edges_closer_to_original(self):
        h, w = 40, 50
        alpha, inv = _build_feather_masks(w, h, feather_px=8)
        original = np.zeros((h, w, 3), dtype=np.uint8)
        inpainted = np.full((h, w, 3), 255, dtype=np.uint8)
        result = _feather_blend(original, inpainted, alpha, inv)
        edge_val = result[0, 0, 0]
        center_val = result[h // 2, w // 2, 0]
        assert edge_val < center_val


class TestBuildCropLookup:
    def test_builds_lookup(self, tmp_path):
        (tmp_path / "frame_000000.jpg").touch()
        (tmp_path / "frame_000001.png").touch()
        lookup = _build_crop_lookup(str(tmp_path))
        assert "frame_000000" in lookup
        assert "frame_000001" in lookup
        assert len(lookup) == 2

    def test_empty_dir(self, tmp_path):
        lookup = _build_crop_lookup(str(tmp_path))
        assert lookup == {}
