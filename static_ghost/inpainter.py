from __future__ import annotations

import shutil
import subprocess
import sys


def check_iopaint() -> None:
    """Verify iopaint CLI is available."""
    if shutil.which("iopaint") is None:
        print("Error: 'iopaint' not found. Install with: pip install iopaint", file=sys.stderr)
        sys.exit(1)


def run(
    input_dir: str,
    mask_path: str,
    output_dir: str,
    device: str = "cpu",
) -> None:
    """Run IOPaint LaMa inpainting on a directory of frames.

    NOTE: CLI flags validated against IOPaint v1.6.0.
    """
    cmd = [
        "iopaint", "run",
        "--model", "lama",
        "--device", device,
        "--image", input_dir,
        "--mask", mask_path,
        "--output", output_dir,
    ]
    subprocess.run(cmd, check=True)
