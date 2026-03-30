#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.export import export_framewise_logs_to_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge VGGT-SLAM framewise .npz logs into a global .ply. "
            "The logged pointclouds are already in world coordinates, so no pose file is needed."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing VGGT framewise .npz logs")
    parser.add_argument("--output-path", required=True, help="Output .ply path")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--nb-neighbors", type=int, default=20)
    parser.add_argument("--std-ratio", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_framewise_logs_to_ply(
        log_dir=Path(args.input_dir),
        output_path=Path(args.output_path),
        voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )
    print(f"Merged {summary.frame_logs} frame logs")
    print(f"Raw points: {summary.raw_points}")
    print(f"Output points: {summary.output_points}")
    print(f"Saved {summary.output_path}")


if __name__ == "__main__":
    main()
