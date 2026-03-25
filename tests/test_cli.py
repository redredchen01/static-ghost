import pytest
from static_ghost.cli import parse_args


def test_parse_remove_basic():
    args = parse_args(["remove", "video.mp4"])
    assert args.command == "remove"
    assert args.video == "video.mp4"
    assert args.region == []
    assert args.device == "cpu"
    assert args.dilation == 5
    assert args.keep_temp is False
    assert args.output is None
    assert args.threshold == 15


def test_parse_remove_with_regions():
    args = parse_args(["remove", "video.mp4", "--region", "10,20,30,40", "--region", "50,60,70,80"])
    assert len(args.region) == 2
    assert args.region[0] == "10,20,30,40"


def test_parse_detect():
    args = parse_args(["detect", "video.mp4"])
    assert args.command == "detect"


def test_parse_remove_all_flags():
    args = parse_args([
        "remove", "video.mp4",
        "--device", "mps",
        "--dilation", "10",
        "--keep-temp",
        "-o", "out.mp4",
        "--threshold", "25",
    ])
    assert args.device == "mps"
    assert args.dilation == 10
    assert args.keep_temp is True
    assert args.output == "out.mp4"
    assert args.threshold == 25


from unittest.mock import patch
from pathlib import Path


def test_remove_manual_region_e2e(sample_video, tmp_path):
    """End-to-end test with mocked fast_remove — copies input frames as 'inpainted' output."""
    output_path = str(tmp_path / "clean.mp4")

    def fake_fast_remove(frames_dir, output_dir, regions, dilation, device="cpu", padding=50):
        import shutil, os
        os.makedirs(output_dir, exist_ok=True)
        for f in Path(frames_dir).glob("*.png"):
            shutil.copy(f, Path(output_dir) / f.name)

    with patch("static_ghost.cli.fast_remove", side_effect=fake_fast_remove):
        with patch("static_ghost.cli.check_iopaint"):
            from static_ghost.cli import parse_args, cmd_remove
            args = parse_args(["remove", str(sample_video), "--region", "10,10,50,50", "-o", output_path])
            cmd_remove(args)

    assert Path(output_path).exists()
    from static_ghost.video_engine import probe
    meta = probe(output_path)
    assert meta["width"] == 320
    assert meta["height"] == 240
