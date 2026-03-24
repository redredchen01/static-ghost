import subprocess
import pytest
from pathlib import Path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Generate a 2-second 320x240 test video with audio using FFmpeg."""
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    return video_path
