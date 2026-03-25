from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from static_ghost.video_engine import probe, extract_sample_frames, extract_all_frames, merge
from static_ghost.detector import Region, detect_static_regions, save_preview
from static_ghost.mask_generator import create_mask
from static_ghost.inpainter import run as run_inpaint, check_iopaint
from static_ghost.fast_inpaint import fast_remove, fast_remove_streamed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="static-ghost", description="Remove static watermarks from videos")
    sub = parser.add_subparsers(dest="command", required=True)

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("video", help="Path to input video file")
    parent.add_argument("--threshold", type=int, default=15, help="Detection sensitivity (default: 15)")

    rm = sub.add_parser("remove", parents=[parent], help="Detect and remove watermarks")
    rm.add_argument("--region", action="append", default=[], help="Manual region x,y,w,h (repeatable)")
    rm.add_argument("--pick", action="store_true", help="Open browser to draw watermark region")
    rm.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Compute device")
    rm.add_argument("--dilation", type=int, default=5, help="Mask dilation pixels (default: 5)")
    rm.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    rm.add_argument("--stream", action="store_true", help="Stream mode: ~95%% less disk usage (no full-frame extraction)")
    rm.add_argument("-o", "--output", help="Output video path")

    sub.add_parser("detect", parents=[parent], help="Detect watermarks only (no removal)")

    pick = sub.add_parser("pick", parents=[parent], help="Open browser to draw watermark region")
    pick.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    pick.add_argument("--dilation", type=int, default=5)
    pick.add_argument("--keep-temp", action="store_true")
    pick.add_argument("-o", "--output", help="Output video path")

    return parser.parse_args(argv)


def _preflight(video_path: str) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"Error: '{tool}' not found in PATH.", file=sys.stderr)
            sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    try:
        probe(video_path)
    except Exception as e:
        print(f"Error: Cannot read video: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_region(region_str: str) -> Region:
    parts = region_str.split(",")
    if len(parts) != 4:
        print(f"Error: Invalid region format '{region_str}'. Expected x,y,w,h", file=sys.stderr)
        sys.exit(1)
    x, y, w, h = (int(p.strip()) for p in parts)
    return Region(x=x, y=y, w=w, h=h, confidence=1.0)


def _warn_disk_space(video_path: str, meta: dict) -> None:
    fps = meta["fps"]
    duration = meta["duration"]
    frame_count = int(fps * duration)
    avg_frame_bytes = meta["width"] * meta["height"] * 3
    estimated_bytes = frame_count * avg_frame_bytes * 2

    stat = os.statvfs(os.path.dirname(os.path.abspath(video_path)))
    available = stat.f_bavail * stat.f_frsize

    if estimated_bytes > available:
        est_gb = estimated_bytes / (1024**3)
        avail_gb = available / (1024**3)
        print(f"Warning: Estimated disk usage {est_gb:.1f} GB, available {avail_gb:.1f} GB", file=sys.stderr)


def _get_regions_interactive(video_path: str, tmp_root: str, threshold: int, use_picker: bool) -> list[Region] | None:
    """Get watermark regions via auto-detect, picker, or user confirmation."""
    if use_picker:
        from static_ghost.picker import pick_region
        print("Extracting a sample frame...")
        samples_dir = os.path.join(tmp_root, "samples")
        sample_paths = extract_sample_frames(video_path, n=1, output_dir=samples_dir)
        if not sample_paths:
            print("Error: Could not extract sample frame.", file=sys.stderr)
            return None
        region = pick_region(sample_paths[0])
        if region:
            print(f"Selected: x={region.x}, y={region.y}, w={region.w}, h={region.h}")
            return [region]
        print("No region selected.")
        return None

    # Auto-detect
    print("Extracting sample frames for detection...")
    samples_dir = os.path.join(tmp_root, "samples")
    sample_paths = extract_sample_frames(video_path, n=30, output_dir=samples_dir)

    regions = detect_static_regions(sample_paths, threshold=threshold)
    if not regions:
        print("No static watermark regions detected.")
        print("Try: --threshold 25-50, or use --pick to draw the region, or --region x,y,w,h")
        return None

    print(f"\nDetected {len(regions)} region(s):")
    for i, r in enumerate(regions):
        print(f"  #{i+1}: x={r.x}, y={r.y}, w={r.w}, h={r.h} (confidence={r.confidence:.2f})")

    preview_path = os.path.join(tmp_root, "preview.png")
    mid = sample_paths[len(sample_paths) // 2]
    save_preview(mid, regions, preview_path)
    print(f"\nPreview: {preview_path}")
    print("Open with: open " + preview_path)

    answer = input("\nProceed with these regions? [Y/n/edit/pick] ").strip().lower()
    if answer == "n":
        return None
    elif answer == "edit":
        region_str = input("Enter regions as x,y,w,h (semicolon-separated): ").strip()
        return [_parse_region(r.strip()) for r in region_str.split(";")]
    elif answer == "pick":
        from static_ghost.picker import pick_region
        region = pick_region(sample_paths[len(sample_paths) // 2])
        return [region] if region else None

    return regions


def cmd_detect(args: argparse.Namespace) -> None:
    _preflight(args.video)

    print("Extracting sample frames...")
    with tempfile.TemporaryDirectory() as tmp:
        samples_dir = os.path.join(tmp, "samples")
        sample_paths = extract_sample_frames(args.video, n=30, output_dir=samples_dir)
        print(f"Extracted {len(sample_paths)} sample frames.")

        regions = detect_static_regions(sample_paths, threshold=args.threshold)

        if not regions:
            print("No static watermark regions detected.")
            print("Try: --threshold 25-50, or use 'static-ghost pick' to draw the region.")
            return

        print(f"\nDetected {len(regions)} region(s):")
        for i, r in enumerate(regions):
            print(f"  #{i+1}: x={r.x}, y={r.y}, w={r.w}, h={r.h} (confidence={r.confidence:.2f})")

        preview_path = os.path.join(os.getcwd(), "preview.png")
        save_preview(sample_paths[len(sample_paths) // 2], regions, preview_path)
        print(f"\nPreview saved: {preview_path}")


def cmd_remove(args: argparse.Namespace) -> None:
    _preflight(args.video)
    check_iopaint()
    meta = probe(args.video)
    _warn_disk_space(args.video, meta)

    tmp_root = tempfile.mkdtemp(prefix="static_ghost_")

    try:
        # Get regions
        if args.region:
            regions = [_parse_region(r) for r in args.region]
            print(f"Using {len(regions)} manual region(s).")
        else:
            use_picker = getattr(args, "pick", False)
            regions = _get_regions_interactive(args.video, tmp_root, args.threshold, use_picker)
            if not regions:
                return

        output_path = args.output or _default_output_path(args.video)

        if getattr(args, "stream", False):
            # Stream mode: no full-frame extraction, ~95% less disk
            print(f"Stream mode: processing directly to {output_path}...")
            try:
                fast_remove_streamed(
                    args.video, output_path, regions,
                    dilation=args.dilation, device=args.device,
                )
            except Exception as e:
                print(f"\nProcessing failed: {e}", file=sys.stderr)
                args.keep_temp = True
                return
        else:
            # Classic mode: extract all frames to disk
            print("Generating mask...")
            mask_path = os.path.join(tmp_root, "mask.png")
            create_mask(meta["width"], meta["height"], regions, mask_path, dilation=args.dilation)

            print("Extracting all frames (this may take a moment)...")
            frames_dir = os.path.join(tmp_root, "frames")
            frame_count = extract_all_frames(args.video, frames_dir)
            print(f"Extracted {frame_count} frames.")

            print("Running IOPaint LaMa inpainting...")
            output_frames_dir = os.path.join(tmp_root, "output")
            os.makedirs(output_frames_dir, exist_ok=True)
            try:
                fast_remove(
                    frames_dir, output_frames_dir, regions,
                    dilation=args.dilation, device=args.device,
                )
            except Exception as e:
                print(f"\nIOPaint failed: {e}", file=sys.stderr)
                print(f"Temp files preserved at: {tmp_root}", file=sys.stderr)
                args.keep_temp = True
                return

            print(f"Merging to {output_path}...")
            try:
                merge(output_frames_dir, args.video, output_path)
            except Exception as e:
                print(f"\nFFmpeg merge failed: {e}", file=sys.stderr)
                print(f"Inpainted frames at: {output_frames_dir}", file=sys.stderr)
                args.keep_temp = True
                return

        print(f"Done! Output: {output_path}")

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        print(f"Temp files preserved at: {tmp_root}", file=sys.stderr)
        args.keep_temp = True
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"Temp files kept at: {tmp_root}")


def cmd_pick(args: argparse.Namespace) -> None:
    """Pick region interactively then run removal."""
    args.region = []
    args.pick = True
    cmd_remove(args)


def _default_output_path(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_stem(p.stem + "_clean"))


def main() -> None:
    args = parse_args()
    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "pick":
        cmd_pick(args)
