#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.config import load_runner_context, validate_context
from hilti_vggt_runner.run import build_vggt_command, run_vggt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT-SLAM on a prepared Hilti sequence.")
    parser.add_argument("--paths", required=True, help="Path config YAML")
    parser.add_argument("--sequence", required=True, help="Sequence config YAML")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow execution without CUDA")
    parser.add_argument("--dry-run", action="store_true", help="Print the VGGT command without running it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_runner_context(args.paths, args.sequence, profile=args.profile)
    validate_context(context)

    if args.dry_run:
        command = build_vggt_command(context)
        print(" ".join(command))
        return

    summary = run_vggt(context, allow_cpu=args.allow_cpu)
    print(f"VGGT command log: {summary.log_path}")
    print(f"Pose log: {summary.poses_path}")
    print(f"Dense frame logs: {summary.dense_log_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
