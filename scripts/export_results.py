#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.config import ensure_layout_dirs, load_runner_context, validate_context, write_resolved_config
from hilti_vggt_runner.export import export_framewise_logs_to_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VGGT-SLAM framewise pointcloud logs to a viewable .ply.")
    parser.add_argument("--paths", required=True, help="Path config YAML")
    parser.add_argument("--sequence", required=True, help="Sequence config YAML")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_runner_context(args.paths, args.sequence, profile=args.profile)
    validate_context(context)
    ensure_layout_dirs(context)
    write_resolved_config(context)

    summary = export_framewise_logs_to_ply(
        log_dir=context.layout.dense_log_dir,
        output_path=context.layout.export_path,
        voxel_size=context.sequence.export.voxel_size,
        nb_neighbors=context.sequence.export.nb_neighbors,
        std_ratio=context.sequence.export.std_ratio,
    )
    print(f"Frame logs merged: {summary.frame_logs}")
    print(f"Raw points: {summary.raw_points}")
    print(f"Output points: {summary.output_points}")
    print(f"Point cloud: {summary.output_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
