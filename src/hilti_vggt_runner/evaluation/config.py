from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


def _expand_envvars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_envvars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_envvars(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at {path}, found {type(loaded).__name__}")
    return _expand_envvars(loaded)


def _as_path(raw_value: str | os.PathLike[str] | None) -> Path | None:
    if raw_value in (None, ""):
        return None
    return Path(raw_value).expanduser().resolve()


AlignmentMode = Literal["rigid_se3", "init_anchor", "sim3_diagnostic"]
AnchorMode = Literal["init_yaw_translation", "init_se3"]


@dataclass(frozen=True)
class ResolvedRunConfig:
    resolved_config_path: Path
    run_name: str
    input_type: str
    profile: str
    frame_manifest_path: Path
    source_metadata_path: Path
    profile_root: Path
    log_path: Path
    dense_log_dir: Path
    export_path: Path
    preview_path: Path


@dataclass(frozen=True)
class TimingConfig:
    absolute_start_time_seconds: float | None
    ignore_initial_seconds: float
    association_tolerance_seconds: float


@dataclass(frozen=True)
class GroundTruthConfig:
    trajectory_txt: Path | None
    init_pose_csv: Path | None
    lookup_run_name: str | None


@dataclass(frozen=True)
class TrajectoryConfig:
    alignment_modes: tuple[AlignmentMode, ...]
    anchor_mode: AnchorMode
    init_anchor_tolerance_seconds: float
    rpe_horizon_seconds: float
    rpe_tolerance_seconds: float


@dataclass(frozen=True)
class FloorplanConfig:
    png_path: Path | None
    base_resolution_m_per_px: float
    eval_resolution_m_per_px: float
    wall_dilation_m: float
    trajectory_corridor_m: float
    wall_match_radius_m: float
    z_min_m: float
    z_max_m: float
    min_points_per_cell: int
    vertical_extent_min_m: float
    prefer_raw_logs: bool


@dataclass(frozen=True)
class EvaluationConfig:
    eval_name: str
    run_name: str | None
    ground_truth: GroundTruthConfig
    timing: TimingConfig
    trajectory: TrajectoryConfig
    floorplan: FloorplanConfig


def load_resolved_run_config(path: str | Path) -> ResolvedRunConfig:
    resolved_path = Path(path).expanduser().resolve()
    cfg = _load_yaml(resolved_path)
    sequence = cfg.get("sequence") or {}
    layout = cfg.get("layout") or {}
    return ResolvedRunConfig(
        resolved_config_path=resolved_path,
        run_name=str(sequence["run_name"]),
        input_type=str((sequence.get("input") or {}).get("type", "unknown")),
        profile=str(cfg.get("profile", "")),
        frame_manifest_path=Path(layout["frame_manifest_path"]).expanduser().resolve(),
        source_metadata_path=Path(layout["source_metadata_path"]).expanduser().resolve(),
        profile_root=Path(layout["profile_root"]).expanduser().resolve(),
        log_path=Path(layout["log_path"]).expanduser().resolve(),
        dense_log_dir=Path(layout["dense_log_dir"]).expanduser().resolve(),
        export_path=Path(layout["export_path"]).expanduser().resolve(),
        preview_path=Path(layout["preview_path"]).expanduser().resolve(),
    )


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path).expanduser().resolve()
    cfg = _load_yaml(config_path)

    gt_cfg = cfg.get("ground_truth") or {}
    timing_cfg = cfg.get("timing") or {}
    traj_cfg = cfg.get("trajectory") or {}
    floor_cfg = cfg.get("floorplan") or {}
    align_modes = tuple(traj_cfg.get("alignment_modes") or ("rigid_se3", "init_anchor"))

    return EvaluationConfig(
        eval_name=str(cfg.get("eval_name") or config_path.stem),
        run_name=str(cfg["run_name"]) if cfg.get("run_name") else None,
        ground_truth=GroundTruthConfig(
            trajectory_txt=_as_path(gt_cfg.get("trajectory_txt")),
            init_pose_csv=_as_path(gt_cfg.get("init_pose_csv")),
            lookup_run_name=(str(gt_cfg["lookup_run_name"]) if gt_cfg.get("lookup_run_name") else None),
        ),
        timing=TimingConfig(
            absolute_start_time_seconds=(
                None
                if timing_cfg.get("absolute_start_time_seconds") in (None, "")
                else float(timing_cfg["absolute_start_time_seconds"])
            ),
            ignore_initial_seconds=float(timing_cfg.get("ignore_initial_seconds", 5.0)),
            association_tolerance_seconds=float(timing_cfg.get("association_tolerance_seconds", 0.05)),
        ),
        trajectory=TrajectoryConfig(
            alignment_modes=align_modes,  # type: ignore[arg-type]
            anchor_mode=str(traj_cfg.get("anchor_mode", "init_yaw_translation")),  # type: ignore[arg-type]
            init_anchor_tolerance_seconds=float(traj_cfg.get("init_anchor_tolerance_seconds", 1.0)),
            rpe_horizon_seconds=float(traj_cfg.get("rpe_horizon_seconds", 1.0)),
            rpe_tolerance_seconds=float(traj_cfg.get("rpe_tolerance_seconds", 0.25)),
        ),
        floorplan=FloorplanConfig(
            png_path=_as_path(floor_cfg.get("png_path")),
            base_resolution_m_per_px=float(floor_cfg.get("base_resolution_m_per_px", 0.01)),
            eval_resolution_m_per_px=float(floor_cfg.get("eval_resolution_m_per_px", 0.05)),
            wall_dilation_m=float(floor_cfg.get("wall_dilation_m", 0.15)),
            trajectory_corridor_m=float(floor_cfg.get("trajectory_corridor_m", 3.0)),
            wall_match_radius_m=float(floor_cfg.get("wall_match_radius_m", 0.15)),
            z_min_m=float(floor_cfg.get("z_min_m", 0.2)),
            z_max_m=float(floor_cfg.get("z_max_m", 2.5)),
            min_points_per_cell=int(floor_cfg.get("min_points_per_cell", 3)),
            vertical_extent_min_m=float(floor_cfg.get("vertical_extent_min_m", 0.5)),
            prefer_raw_logs=bool(floor_cfg.get("prefer_raw_logs", True)),
        ),
    )


def validate_evaluation_config(resolved: ResolvedRunConfig, evaluation: EvaluationConfig) -> None:
    problems: list[str] = []

    if evaluation.run_name and evaluation.run_name != resolved.run_name:
        problems.append(
            f"Evaluation run_name {evaluation.run_name!r} does not match resolved config run_name {resolved.run_name!r}"
        )

    if not resolved.log_path.is_file():
        problems.append(f"Estimated poses.txt not found: {resolved.log_path}")
    if not resolved.frame_manifest_path.is_file():
        problems.append(f"Frame manifest not found: {resolved.frame_manifest_path}")

    if evaluation.ground_truth.trajectory_txt is not None and not evaluation.ground_truth.trajectory_txt.is_file():
        problems.append(f"Ground-truth trajectory not found: {evaluation.ground_truth.trajectory_txt}")
    if evaluation.ground_truth.init_pose_csv is not None and not evaluation.ground_truth.init_pose_csv.is_file():
        problems.append(f"Initial pose CSV not found: {evaluation.ground_truth.init_pose_csv}")
    if evaluation.floorplan.png_path is not None and not evaluation.floorplan.png_path.is_file():
        problems.append(f"Floorplan PNG not found: {evaluation.floorplan.png_path}")

    if resolved.input_type == "mp4" and evaluation.ground_truth.trajectory_txt and evaluation.timing.absolute_start_time_seconds is None:
        problems.append("MP4 trajectory evaluation requires timing.absolute_start_time_seconds")

    if evaluation.timing.ignore_initial_seconds < 0:
        problems.append("timing.ignore_initial_seconds must be non-negative")
    if evaluation.timing.association_tolerance_seconds <= 0:
        problems.append("timing.association_tolerance_seconds must be greater than 0")
    if evaluation.trajectory.rpe_horizon_seconds <= 0:
        problems.append("trajectory.rpe_horizon_seconds must be greater than 0")
    if evaluation.trajectory.rpe_tolerance_seconds <= 0:
        problems.append("trajectory.rpe_tolerance_seconds must be greater than 0")
    if evaluation.trajectory.init_anchor_tolerance_seconds <= 0:
        problems.append("trajectory.init_anchor_tolerance_seconds must be greater than 0")
    if evaluation.floorplan.base_resolution_m_per_px <= 0:
        problems.append("floorplan.base_resolution_m_per_px must be greater than 0")
    if evaluation.floorplan.eval_resolution_m_per_px <= 0:
        problems.append("floorplan.eval_resolution_m_per_px must be greater than 0")
    if evaluation.floorplan.eval_resolution_m_per_px < evaluation.floorplan.base_resolution_m_per_px:
        problems.append("floorplan.eval_resolution_m_per_px cannot be finer than the base floorplan resolution")
    if evaluation.floorplan.wall_dilation_m < 0:
        problems.append("floorplan.wall_dilation_m must be non-negative")
    if evaluation.floorplan.wall_match_radius_m < 0:
        problems.append("floorplan.wall_match_radius_m must be non-negative")
    if evaluation.floorplan.trajectory_corridor_m <= 0:
        problems.append("floorplan.trajectory_corridor_m must be greater than 0")
    if evaluation.floorplan.min_points_per_cell <= 0:
        problems.append("floorplan.min_points_per_cell must be greater than 0")
    if evaluation.floorplan.z_max_m <= evaluation.floorplan.z_min_m:
        problems.append("floorplan.z_max_m must be greater than floorplan.z_min_m")
    if evaluation.floorplan.vertical_extent_min_m <= 0:
        problems.append("floorplan.vertical_extent_min_m must be greater than 0")

    if problems:
        raise ValueError("\n".join(problems))
