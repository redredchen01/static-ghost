from pathlib import Path
from static_ghost.video_engine import probe, extract_sample_frames, extract_all_frames, merge


def test_probe_returns_metadata(sample_video):
    meta = probe(str(sample_video))
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 10.0
    assert meta["duration"] > 0
    assert "codec" in meta
    assert "audio_codec" in meta


def test_extract_sample_frames(sample_video, tmp_path):
    output_dir = str(tmp_path / "samples")
    paths = extract_sample_frames(str(sample_video), n=5, output_dir=output_dir)
    assert len(paths) == 5
    for p in paths:
        assert Path(p).exists()
        assert p.endswith(".png")


def test_extract_all_frames(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames")
    count = extract_all_frames(str(sample_video), output_dir=output_dir)
    assert count == 20  # 2 seconds * 10 fps
    pngs = sorted(Path(output_dir).glob("*.png"))
    assert len(pngs) == count


def test_merge_produces_video(sample_video, tmp_path):
    frames_dir = str(tmp_path / "frames")
    extract_all_frames(str(sample_video), frames_dir)

    output_path = str(tmp_path / "output.mp4")
    result = merge(frames_dir, str(sample_video), output_path)
    assert Path(result).exists()

    meta = probe(result)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["audio_codec"] is not None  # audio preserved
