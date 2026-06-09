"""
TopoTremor batch comparison.

Runs the landmark-free optical-flow + persistent-homology pipeline over several
videos NON-INTERACTIVELY and prints a comparison table. This is where the
framework claim starts becoming real: tremor should show a higher
`tremor_power_ratio` AND a longer `h1_lifetime` than steady / non-cyclic motion.

Usage:
    python compare_videos.py                 # full-frame ROI on the default set
    python compare_videos.py --roi           # interactively draw an ROI per video
    python compare_videos.py a.mp4 b.mp4      # custom video list
    python compare_videos.py --plots         # also save per-video diagnostic plots

Outputs:
    outputs/comparison.csv
    (with --plots) outputs/<stem>_*.png for each video
"""

import argparse
import csv
import os

import cv2
import numpy as np

# Reuse the exact pipeline functions from the MVP so the two scripts can never
# drift apart in their analysis logic.
from topotremor_mvp import (
    analyze_signal,
    extract_optical_flow_signal,
    save_plots,
    select_roi_first_frame,
)


# Default battery of test videos (place them in videos/).
DEFAULT_VIDEOS = [
    "steady.mp4",
    "fake_tremor.mp4",
    "voluntary_wave.mp4",
    "bad_lighting.mp4",
    "phone_vibration.mp4",
]

VIDEO_DIR = "videos"
OUTPUT_DIR = "outputs"

# Quality gate: a run is only "valid" if tracking held up across the clip.
MIN_MEDIAN_TRACKED = 25
MIN_TRACKED_FLOOR = 10


def full_frame_roi(video_path):
    """Whole-frame ROI so the batch can run without any GUI interaction."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read video: {video_path}")
    h, w = frame.shape[:2]
    return 0, 0, w, h


def process_one(video_path, interactive=False, make_plots=False):
    """
    Run the full pipeline on one video.

    Returns a row dict. `valid` is False (with `error` populated) when the clip
    is too short, untrackable, or tracking collapsed mid-way.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    row = {
        "video": os.path.basename(video_path),
        "peak_frequency": None,
        "tremor_power_ratio": None,
        "h1_lifetime": None,
        "tracked_points": None,
        "valid": False,
        "error": "",
    }

    if not os.path.exists(video_path):
        row["error"] = "missing file"
        return row

    try:
        roi = select_roi_first_frame(video_path) if interactive else full_frame_roi(video_path)
        dx, dy, tracked_counts, fs = extract_optical_flow_signal(video_path, roi)

        # Analyze horizontal micro-motion (rhythm-preserving), matching the MVP.
        signal = dx
        analysis = analyze_signal(signal, fs)

        median_tracked = float(np.median(tracked_counts))
        min_tracked = float(np.min(tracked_counts))

        row["peak_frequency"] = round(analysis["peak_frequency"], 2)
        row["tremor_power_ratio"] = round(analysis["tremor_power_ratio"], 4)
        row["h1_lifetime"] = round(analysis["h1_lifetime"], 4)
        row["tracked_points"] = int(median_tracked)

        # Tracking is "valid" only if it stayed healthy for the whole clip.
        tracking_ok = median_tracked >= MIN_MEDIAN_TRACKED and min_tracked >= MIN_TRACKED_FLOOR
        row["valid"] = bool(tracking_ok)
        if not tracking_ok:
            row["error"] = f"weak tracking (median={median_tracked:.0f}, min={min_tracked:.0f})"

        if make_plots:
            prefix = os.path.join(OUTPUT_DIR, stem)
            save_plots(signal, analysis, tracked_counts, fs, prefix)

    except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the batch
        row["error"] = str(exc)

    return row


def fmt(value):
    return "—" if value is None else str(value)


def print_table(rows):
    headers = ["video", "peak_freq", "power_ratio", "h1_lifetime", "tracked", "valid"]
    widths = [22, 9, 11, 11, 7, 5]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print()
    print(line(headers))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line([
            r["video"],
            fmt(r["peak_frequency"]),
            fmt(r["tremor_power_ratio"]),
            fmt(r["h1_lifetime"]),
            fmt(r["tracked_points"]),
            "yes" if r["valid"] else "no",
        ]))
        if r["error"]:
            print(f"    └─ {r['error']}")
    print()


def write_csv(rows, path):
    fields = ["video", "peak_frequency", "tremor_power_ratio", "h1_lifetime", "tracked_points", "valid", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def summarize(rows):
    """Print the one comparison that actually matters for the milestone."""
    valid = [r for r in rows if r["valid"]]
    if len(valid) < 2:
        print("Need at least two valid videos to compare topological signatures.")
        return

    ranked = sorted(valid, key=lambda r: r["h1_lifetime"], reverse=True)
    print("Ranked by H1 loop lifetime (strongest cyclic structure first):")
    for r in ranked:
        print(f"  {r['video']:<22}  H1={r['h1_lifetime']:<8}  power_ratio={r['tremor_power_ratio']}")
    print()
    print(f"Strongest cyclic signature: {ranked[0]['video']}")
    print("Milestone passes if a tremor clip ranks above steady/voluntary clips.")
    print()


def main():
    parser = argparse.ArgumentParser(description="TopoTremor batch comparison")
    parser.add_argument("videos", nargs="*", help="Video files (in videos/ or absolute paths).")
    parser.add_argument("--roi", action="store_true", help="Interactively select an ROI per video.")
    parser.add_argument("--plots", action="store_true", help="Save per-video diagnostic plots.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    names = args.videos if args.videos else DEFAULT_VIDEOS
    paths = [v if os.path.isabs(v) or os.path.dirname(v) else os.path.join(VIDEO_DIR, v) for v in names]

    rows = []
    for p in paths:
        print(f"Processing {os.path.basename(p)} ...")
        rows.append(process_one(p, interactive=args.roi, make_plots=args.plots))

    print_table(rows)

    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")
    print()

    summarize(rows)


if __name__ == "__main__":
    main()
