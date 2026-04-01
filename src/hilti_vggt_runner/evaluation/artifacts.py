from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import EvaluationConfig, ResolvedRunConfig


@dataclass(frozen=True)
class RunArtifacts:
    resolved: ResolvedRunConfig
    evaluation: EvaluationConfig
    output_dir: Path
    plots_dir: Path
    metrics_json_path: Path
    metrics_csv_path: Path
    matched_poses_csv_path: Path
    report_path: Path


def load_run_artifacts(resolved: ResolvedRunConfig, evaluation: EvaluationConfig) -> RunArtifacts:
    output_dir = resolved.profile_root / "evaluation" / evaluation.eval_name
    return RunArtifacts(
        resolved=resolved,
        evaluation=evaluation,
        output_dir=output_dir,
        plots_dir=output_dir / "plots",
        metrics_json_path=output_dir / "metrics.json",
        metrics_csv_path=output_dir / "metrics.csv",
        matched_poses_csv_path=output_dir / "matched_poses.csv",
        report_path=output_dir / "report.md",
    )


def ensure_artifact_dirs(artifacts: RunArtifacts) -> None:
    for path in [artifacts.output_dir, artifacts.plots_dir]:
        path.mkdir(parents=True, exist_ok=True)
