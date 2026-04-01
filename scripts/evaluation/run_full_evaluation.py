#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hilti_vggt_runner.evaluation.config import (
    EvaluationConfig,
    FloorplanConfig,
    GroundTruthConfig,
    load_evaluation_config,
    load_resolved_run_config,
    validate_evaluation_config,
)
from hilti_vggt_runner.evaluation.pipeline import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory and floorplan evaluation for a completed VGGT run.")
    parser.add_argument("--resolved-config", required=True, help="Path to the reconstruction resolved_config.yaml")
    parser.add_argument("--eval-config", required=True, help="Path to the evaluation config YAML")
    parser.add_argument("--trajectory-only", action="store_true", help="Skip floorplan evaluation")
    parser.add_argument("--floorplan-only", action="store_true", help="Skip full GT trajectory evaluation")
    return parser.parse_args()


def _apply_mode_overrides(config: EvaluationConfig, *, trajectory_only: bool, floorplan_only: bool) -> EvaluationConfig:
    if trajectory_only and floorplan_only:
        raise ValueError("Choose at most one of --trajectory-only or --floorplan-only")
    if trajectory_only:
        return replace(
            config,
            floorplan=replace(config.floorplan, png_path=None),
        )
    if floorplan_only:
        return replace(
            config,
            ground_truth=replace(config.ground_truth, trajectory_txt=None),
        )
    return config


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    resolved = load_resolved_run_config(args.resolved_config)
    evaluation = load_evaluation_config(args.eval_config)
    evaluation = _apply_mode_overrides(
        evaluation,
        trajectory_only=args.trajectory_only,
        floorplan_only=args.floorplan_only,
    )
    validate_evaluation_config(resolved, evaluation)
    result = run_evaluation(resolved, evaluation)

    print(f"Evaluated run: {resolved.run_name}")
    print(f"Evaluation output: {result.artifacts.output_dir}")
    print(f"Metrics JSON: {result.artifacts.metrics_json_path}")
    print(f"Metrics CSV: {result.artifacts.metrics_csv_path}")
    print(f"Report: {result.artifacts.report_path}")
    if "rigid_se3" in result.trajectory_evaluations:
        metrics = result.trajectory_evaluations["rigid_se3"].metrics
        print(f"ATE XY RMSE [m]: {metrics['ate_xy_m_rmse']:.4f}")
        print(f"ATE 3D RMSE [m]: {metrics['ate_3d_m_rmse']:.4f}")
    if result.floorplan_evaluation is not None:
        metrics = result.floorplan_evaluation.metrics
        print(f"Wall precision: {_format_metric(metrics['wall_precision'])}")
        print(f"Wall recall: {_format_metric(metrics['wall_recall'])}")
        print(f"Wall distance p95 [m]: {_format_metric(metrics['wall_distance_p95_m'])}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
