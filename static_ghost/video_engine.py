from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path


def probe(video_path: str) -> dict:
    """Extract video metadata via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    streams = data["streams"]
    fmt = data.get("format", {})

    video_stream = next(s for s in streams if s["codec_type"] == "video")
    audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)

    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    fps = float(Fraction(r_frame_rate))

    duration = float(video_stream.get("duration", 0))
    if duration == 0:
        duration = float(fmt.get("duration", 0))

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration": duration,
        "codec": video_stream["codec_name"],
        "audio_codec": audio_stream["codec_name"] if audio_stream else None,
    }


def extract_sample_frames(video_path: str, n: int, output_dir: str) -> list[str]:
    """Extract n evenly-spaced frames from video as PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    meta = probe(video_path)
    duration = meta["duration"]

    paths = []
    for i in range(n):
        timestamp = (duration / (n + 1)) * (i + 1)
        out_path = os.path.join(output_dir, f"sample_{i:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "1",
                out_path,
            ],
            capture_output=True,
        )
        if os.path.isfile(out_path):
            paths.append(out_path)
    return paths


def extract_all_frames(video_path: str, output_dir: str) -> int:
    """Extract all frames as numbered PNGs. Returns total frame count."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%06d.png")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-q:v", "1",
            pattern,
        ],
        capture_output=True,
        check=True,
    )
    return len([f for f in os.listdir(output_dir) if f.endswith(".png")])


def merge(frames_dir: str, video_path: str, output_path: str) -> str:
    """Merge image frames + original audio into output video."""
    meta = probe(video_path)
    fps = meta["fps"]

    # Auto-detect frame format (PNG or JPEG)
    sample_files = os.listdir(frames_dir)
    if any(f.endswith(".png") for f in sample_files):
        pattern = os.path.join(frames_dir, "frame_%06d.png")
    else:
        pattern = os.path.join(frames_dir, "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-i", video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
