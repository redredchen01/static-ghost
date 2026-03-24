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
