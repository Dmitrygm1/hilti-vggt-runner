#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.config import ProfileName, load_runner_context, validate_context
from hilti_vggt_runner.prepare import prepare_profile_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Hilti equirectangular MP4 frames for VGGT-SLAM.")
    parser.add_argument("--paths", required=True, help="Path config YAML")
    parser.add_argument("--sequence", required=True, help="Sequence config YAML")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--force", action="store_true", help="Re-extract frames and rebuild subsets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_runner_context(args.paths, args.sequence, profile=args.profile)
    validate_context(context)
    summary = prepare_profile_inputs(context, force=args.force)

    print(f"Prepared profile: {args.profile}")
    print(f"Source video: {context.sequence.source_mp4}")
    print(
        "Video metadata: "
        f"{summary.metadata.width}x{summary.metadata.height}, "
        f"{summary.metadata.source_fps:.2f} fps, "
        f"{summary.metadata.frame_count} frames"
    )
    print(
        "Extraction: "
        f"stride={summary.stride}, "
        f"target_fps={context.sequence.extraction.sample_fps:.2f}, "
        f"actual_fps={summary.actual_sample_fps:.2f}, "
        f"frames={summary.extracted_frame_count}"
    )
    print(f"Canonical frames: {context.layout.frames_dir}")
    if context.profile == "smoke":
        print(f"Smoke subset: {context.layout.smoke_frames_dir}")
    print(f"Manifest: {context.layout.frame_manifest_path}")


if __name__ == "__main__":
    main()
