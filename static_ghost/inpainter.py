from __future__ import annotations

import shutil
import sys

from static_ghost.fast_inpaint import _run_iopaint


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
    """Run IOPaint LaMa inpainting on a directory of frames."""
    _run_iopaint(input_dir, mask_path, output_dir, device)
