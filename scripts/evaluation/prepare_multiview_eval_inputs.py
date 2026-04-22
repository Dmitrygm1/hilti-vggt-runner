#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.evaluation.multiview import derive_multiview_evaluation_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a one-view-per-physical-frame trajectory for evaluating a multiview VGGT run."
    )
    parser.add_argument("--resolved-config", required=True, help="Path to the reconstruction resolved_config.yaml")
    parser.add_argument("--view-index", type=int, default=None, help="View index to keep. Defaults to the manifest's eval-primary view.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = derive_multiview_evaluation_inputs(
        Path(args.resolved_config),
        view_index=args.view_index,
    )
    print(f"Derived evaluation inputs: {summary.output_root}")
    print(f"Resolved config: {summary.resolved_config_path}")
    print(f"Poses: {summary.poses_path}")
    print(f"Manifest: {summary.manifest_path}")
    print(f"Selected view index: {summary.selected_view_index}")
    print(f"Selected emitted frames: {summary.selected_frame_count}")
    print(f"Physical frames covered: {summary.physical_frame_count}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
