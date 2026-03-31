#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.rosbag import RosbagStitchConfig, extract_rosbag_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract stitched equirectangular frames from Hilti ROS2 bag(s).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bag", help="Path to a single rosbag.db3 file")
    group.add_argument("--data_dir", help="Root directory containing .../rosbag/rosbag.db3 runs")
    parser.add_argument("--yaml", required=True, help="Path to kalibr_imucam_chain.yaml")
    parser.add_argument("--out_dir", default=None, help="Output frame directory or output root with --data_dir")
    parser.add_argument("--mask0", default=None, help="Optional grayscale camera mask for cam0")
    parser.add_argument("--mask1", default=None, help="Optional grayscale camera mask for cam1")
    parser.add_argument("--stride", type=int, default=10, help="Save every Nth synchronized pair")
    parser.add_argument("--max_frames", type=int, default=0, help="Max stitched frames per bag (0 = all)")
    parser.add_argument("--sphere_m", type=float, default=10.0, help="Virtual projection sphere radius in meters")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for output frames")
    parser.add_argument("--rotate_180", action="store_true", help="Rotate the stitched panorama by 180 degrees")
    parser.add_argument("--sync_tolerance_ns", type=int, default=5_000_000, help="Camera sync tolerance in ns")
    parser.add_argument("--topic0", default="/cam0/image_raw/compressed", help="First camera topic")
    parser.add_argument("--topic1", default="/cam1/image_raw/compressed", help="Second camera topic")
    return parser.parse_args()


def _bags_to_process(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.bag:
        if args.out_dir is None:
            raise SystemExit("--out_dir is required when using --bag")
        return [(Path(args.bag), Path(args.out_dir))]

    data_root = Path(args.data_dir)
    bags: list[tuple[Path, Path]] = []
    for bag_path in sorted(data_root.rglob("rosbag.db3")):
        run_dir = bag_path.parent.parent
        if args.out_dir:
            out_dir = Path(args.out_dir) / run_dir.relative_to(data_root)
        else:
            out_dir = run_dir / "equirect_frames"
        bags.append((bag_path, out_dir))
    return bags


def main() -> None:
    args = parse_args()
    bags = _bags_to_process(args)
    print(f"Found {len(bags)} bag(s)")

    for index, (bag_path, out_dir) in enumerate(bags, start=1):
        print(f"\n[{index}/{len(bags)}] {bag_path}")
        summary = extract_rosbag_frames(
            RosbagStitchConfig(
                bag_path=bag_path,
                out_dir=out_dir,
                frame_manifest_path=out_dir.parent / "frame_manifest.csv",
                stitch_summary_path=out_dir.parent / "stitch_summary.yaml",
                preview_path=out_dir.parent / "frame_preview.jpg",
                calibration_yaml=Path(args.yaml),
                mask0=Path(args.mask0) if args.mask0 else None,
                mask1=Path(args.mask1) if args.mask1 else None,
                jpeg_quality=args.jpeg_quality,
                sphere_m=args.sphere_m,
                stride=args.stride,
                max_frames=args.max_frames,
                rotate_180=args.rotate_180,
                sync_tolerance_ns=args.sync_tolerance_ns,
                topic0=args.topic0,
                topic1=args.topic1,
            )
        )
        print(
            f"Saved {summary.extracted_frame_count} frames at "
            f"{summary.output_width}x{summary.output_height} to {out_dir}"
        )


if __name__ == "__main__":
    main()
