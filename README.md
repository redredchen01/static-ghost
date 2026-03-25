# Static Ghost

Remove static watermarks from videos using LaMa inpainting.

Takes a video with a fixed-position watermark (TV logo, site branding, corner text), detects or lets you specify the watermark location, and removes it frame-by-frame using the [LaMa](https://github.com/advimman/lama) inpainting model via [IOPaint](https://github.com/Sanster/IOPaint).

## How it works

```
Input video → Extract frames → Crop watermark region → LaMa inpaint → Paste back → Reassemble video
              (FFmpeg)         (multiprocess)          (IOPaint)       (multiprocess)  (FFmpeg)
```

The **crop-and-paste optimization** only processes the small watermark region instead of the full frame — typically 10-15x fewer pixels, making a 10-minute 1080p video processable in ~2 hours on CPU instead of 20+.

## Install

**Prerequisites:**
```bash
brew install ffmpeg        # or your package manager
pip install iopaint        # LaMa inpainting engine
```

**Install static-ghost:**
```bash
git clone https://github.com/redredchen01/static-ghost.git
cd static-ghost
pip install -e ".[dev]"
```

> **macOS note:** If `iopaint` is not in PATH after install:
> ```bash
> export PATH="$HOME/Library/Python/3.9/bin:$PATH"
> ```

## Usage

### Quick start — draw the watermark region

```bash
static-ghost pick video.mp4 --dilation 15 --device mps -o video_clean.mp4
```

Opens your browser with a frame from the video. Draw a rectangle around the watermark, click confirm, and it runs the full removal pipeline.

### Specify coordinates directly

```bash
static-ghost remove video.mp4 --region 1400,920,520,160 --dilation 15 --device mps
```

Coordinates are `x,y,width,height` from the top-left corner. Multiple watermarks:

```bash
static-ghost remove video.mp4 \
  --region 1400,920,520,160 \
  --region 20,15,200,60 \
  --dilation 15
```

### Auto-detect watermark

```bash
static-ghost detect video.mp4
```

Uses multi-frame differencing to find regions that stay static across the video. Works best on opaque, high-contrast watermarks. For semi-transparent watermarks, raise the threshold:

```bash
static-ghost detect video.mp4 --threshold 35
```

### Full auto pipeline

```bash
static-ghost remove video.mp4 --device mps
```

Auto-detects → shows preview → asks for confirmation → removes.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--region x,y,w,h` | — | Watermark bounding box (repeatable) |
| `--pick` | — | Open browser to draw region |
| `--dilation N` | 5 | Expand mask by N pixels (use 10-15 for logos) |
| `--device cpu\|mps` | cpu | `mps` = Apple Metal GPU, ~2x faster |
| `--threshold N` | 15 | Detection sensitivity (higher = more permissive) |
| `--keep-temp` | — | Keep extracted frames for debugging |
| `-o PATH` | `{name}_clean.mp4` | Output path |

## Performance

Crop-and-paste mode (default). Times for 1080p 30fps video:

| Video length | Frames | CPU | MPS (Apple Metal) |
|-------------|--------|-----|-------------------|
| 30 sec | 900 | ~8 min | ~4 min |
| 10 min | 18,000 | ~2.5 hr | ~1.5 hr |
| 30 min | 54,000 | ~7.5 hr | ~4 hr |

Actual times depend on watermark size and system load. Use `--device mps` on macOS for best performance.

## Tips

- **Always test on 30 seconds first.** Extract a clip with `ffmpeg -i input.mp4 -t 30 -c copy test.mp4` and verify the result before committing to a full video.
- **Be generous with region size.** It's better to over-cover the watermark area than to leave edges visible. Add 50-100px padding beyond the visible watermark.
- **Dilation matters.** Default is 5px. For large logos with solid color blocks, use 15-20.
- **Semi-transparent watermarks** are harder. Auto-detect may fail — use `--pick` or `--region` to specify manually.

## Project structure

```
static_ghost/
├── cli.py              # CLI entry point and flow orchestration
├── video_engine.py     # FFmpeg: probe, extract frames, merge
├── detector.py         # Multi-frame differencing watermark detection
├── mask_generator.py   # Coordinates → binary mask PNG
├── fast_inpaint.py     # Crop-inpaint-paste with multiprocessing
├── inpainter.py        # IOPaint CLI wrapper
└── picker.py           # Browser-based interactive region selector
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

20 tests covering all modules. Tests mock IOPaint so they run without it installed.

---

# Agent Integration Guide

This section is for AI coding agents (Claude Code, Cursor, Copilot, etc.) that need to use static-ghost programmatically.

## When to use this tool

- User wants to remove a watermark/logo from a **video** (not an image)
- The watermark is **static** — fixed position across all frames
- Examples: TV station logos, website branding, "SAMPLE" overlays

**Do NOT use for:** single images, moving/animated watermarks, subtitles (use subtitle extraction instead).

## Python API

```python
from static_ghost.video_engine import probe, extract_sample_frames, extract_all_frames, merge
from static_ghost.detector import Region, detect_static_regions, save_preview
from static_ghost.mask_generator import create_mask
from static_ghost.fast_inpaint import fast_remove
from static_ghost.inpainter import check_iopaint
```

### Step 1: Probe

```python
meta = probe(video_path)
# Returns: {"width": 1920, "height": 1080, "fps": 30.0, "duration": 637.1, "codec": "h264", "audio_codec": "aac"}
total_frames = int(meta["fps"] * meta["duration"])
```

### Step 2: Get watermark coordinates

**Option A — User provides coordinates:**
```python
regions = [Region(x=1400, y=920, w=520, h=160, confidence=1.0)]
```

**Option B — Auto-detect:**
```python
import tempfile
tmp = tempfile.mkdtemp()
sample_paths = extract_sample_frames(video_path, n=30, output_dir=tmp)
regions = detect_static_regions(sample_paths, threshold=15)
# If empty, try threshold=25, 35, 50
# If still empty, fall back to visual inspection or ask user
```

**Option C — Visual inspection (when agent can see images):**
```python
# Extract sample frames
paths = extract_sample_frames(video_path, n=5, output_dir=tmp)
# Read frames with vision tool, inspect corners for watermarks
# Crop suspected area to verify:
import cv2
img = cv2.imread(paths[0])
crop = img[h-150:h, w-500:w]  # bottom-right corner
cv2.imwrite("/tmp/corner.png", crop)
# Estimate coordinates from visual inspection
```

### Step 3: Test on 30-second clip

```python
import subprocess
subprocess.run(["ffmpeg", "-y", "-i", video_path, "-t", "30", "-c", "copy", "/tmp/test_30s.mp4"], capture_output=True, check=True)
```

Run removal on clip, verify output visually, then proceed to full video.

### Step 4: Run removal

```python
from static_ghost.cli import parse_args, cmd_remove

args = parse_args([
    "remove", video_path,
    "--region", "1400,920,520,160",
    "--dilation", "15",
    "--device", "mps",          # "cpu" if no Metal GPU
    "-o", output_path,
])
cmd_remove(args)
```

**For long videos, run in background** (if your environment supports it) and check progress:
```python
import os
# Count output frames in temp dir
tmp_dirs = [d for d in os.listdir("/var/folders/...") if d.startswith("static_ghost_")]
# Compare against total_frames for progress
```

### Step 5: Verify

```python
orig_meta = probe(video_path)
clean_meta = probe(output_path)
assert orig_meta["width"] == clean_meta["width"]
assert orig_meta["height"] == clean_meta["height"]
assert abs(orig_meta["duration"] - clean_meta["duration"]) < 1.0
assert clean_meta["audio_codec"] is not None
# Visually verify sample frames from output
```

## Decision tree for agents

```
User wants watermark removed from video
│
├─ User provided coordinates? → Use them directly
├─ Auto-detect finds regions? → Show to user for confirmation
├─ Auto-detect fails?
│   ├─ Agent has vision? → Extract frames, inspect corners, estimate coords
│   └─ Agent has no vision? → Ask user for coordinates or use --pick
│
├─ Test on 30s clip
│   ├─ Watermark gone? → Run full video
│   ├─ Partially visible? → Increase region size / dilation, re-test
│   └─ Artifacts? → Reduce dilation, re-test
│
└─ Run full video (background for >5 min videos)
```

## Time estimation

Before running the full video, benchmark 5 frames to estimate total time:

```python
import time
test_paths = extract_sample_frames(video_path, n=5, output_dir="/tmp/bench_in")
os.makedirs("/tmp/bench_out", exist_ok=True)
start = time.time()
fast_remove("/tmp/bench_in", "/tmp/bench_out", regions, dilation=15, device="mps")
per_frame = (time.time() - start) / 5
est_minutes = per_frame * total_frames / 60
print(f"Estimated: {est_minutes:.0f} minutes")
```

## Common pitfalls

| Mistake | Fix |
|---------|-----|
| Region too small | Add 50-100px padding beyond visible watermark edges |
| Dilation too low for large logos | Use `--dilation 15` for logos with solid color blocks |
| Running full video without testing | Always test on 30s clip first |
| Forgetting `--device mps` on macOS | 2x speed improvement for free |
| Auto-detect on semi-transparent watermarks | Will likely fail — use manual coordinates |
| Not checking disk space | 1080p 10min ≈ 40-80GB temp space |
