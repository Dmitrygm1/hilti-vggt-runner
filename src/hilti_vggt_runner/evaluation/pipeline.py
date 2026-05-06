from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import RunArtifacts, ensure_artifact_dirs, load_run_artifacts
from .config import EvaluationConfig, ResolvedRunConfig
from .floorplan import FloorplanEvaluationResult, evaluate_floorplan_consistency, load_floorplan
from .plotting import (
    plot_best_fit_trajectory,
    plot_floorplan_overlay_eval_resolution,
    plot_rpe_rotation_timeseries,
    plot_rpe_translation_timeseries,
    plot_translation_error_histogram,
    plot_translation_error_timeseries,
    plot_wall_consistency_overlay,
)
from .pointcloud import AlignedPointcloudSummary, write_aligned_pointcloud
from .poses import EstimatedPoseLoadSummary, load_estimated_pose_sequence, load_init_pose, load_tum_pose_sequence
from .report import write_markdown_report, write_matched_poses_csv, write_metrics_csv, write_metrics_json
from .trajectory import SequenceHealth, TrajectoryEvaluation, compute_sequence_health, evaluate_trajectory_modes
from ..views import read_frame_manifest


@dataclass(frozen=True)
class EvaluationResult:
    artifacts: RunArtifacts
    estimated_summary: EstimatedPoseLoadSummary
    estimated_health: SequenceHealth
    trajectory_evaluations: dict[str, TrajectoryEvaluation]
    floorplan_evaluation: FloorplanEvaluationResult | None
    payload: dict[str, Any]


def _collect_plot_paths(artifacts: RunArtifacts) -> dict[str, Path]:
    return {
        "trajectory_best_fit": artifacts.plots_dir / "trajectory_xy_best_fit.png",
        "translation_error_3d_vs_time": artifacts.plots_dir / "translation_error_3d_vs_time.png",
        "translation_error_xy_vs_time": artifacts.plots_dir / "translation_error_xy_vs_time.png",
        "translation_error_3d_histogram": artifacts.plots_dir / "translation_error_3d_histogram.png",
        "translation_error_xy_histogram": artifacts.plots_dir / "translation_error_xy_histogram.png",
        "rpe_translation_m_vs_index": artifacts.plots_dir / "rpe_translation_m_vs_index.png",
        "rpe_rotation_deg_vs_index": artifacts.plots_dir / "rpe_rotation_deg_vs_index.png",
        "floorplan_overlay": artifacts.plots_dir / "floorplan_overlay.png",
        "pointcloud_floorplan_overlay": artifacts.plots_dir / "pointcloud_floorplan_overlay.png",
        "wall_consistency_overlay": artifacts.plots_dir / "wall_consistency_overlay.png",
    }


def _format_metric(value: Any, *, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return "n/a"
        return f"{value:.{precision}f}"
    return str(value)


def _assert_single_view_manifest(resolved: ResolvedRunConfig) -> None:
    records = read_frame_manifest(resolved.frame_manifest_path)
    view_indices = {record.view_index for record in records}
    if len(view_indices) <= 1:
        return
    raise RuntimeError(
        "This resolved config still points at a multiview manifest.\n"
        "Derive one evaluation trajectory first, for example:\n"
        "python scripts/evaluation/prepare_multiview_eval_inputs.py "
        f"--resolved-config {resolved.resolved_config_path}\n"
        "Then evaluate using the derived resolved_config.yaml under "
        "<profile_root>/evaluation_inputs/view_XX/."
    )


def _write_aligned_pointclouds(
    resolved: ResolvedRunConfig,
    artifacts: RunArtifacts,
    trajectory_evaluations: dict[str, TrajectoryEvaluation],
    floorplan_evaluation: FloorplanEvaluationResult | None,
) -> dict[str, AlignedPointcloudSummary]:
    if not resolved.export_path.is_file():
        return {}

    transforms = {
        mode: evaluation_result.transform
        for mode, evaluation_result in trajectory_evaluations.items()
        if evaluation_result.transform is not None
    }
    if floorplan_evaluation is not None and "init_anchor" not in transforms:
        transforms["init_anchor"] = floorplan_evaluation.anchor.transform

    summaries: dict[str, AlignedPointcloudSummary] = {}
    for mode, transform in transforms.items():
        if transform is None:
            continue
        summaries[mode] = write_aligned_pointcloud(
            mode=mode,
            source_path=resolved.export_path,
            output_path=artifacts.aligned_pointcloud_dir / f"aligned_{mode}.ply",
            transform=transform,
        )
    return summaries


def run_evaluation(resolved: ResolvedRunConfig, evaluation: EvaluationConfig) -> EvaluationResult:
    _assert_single_view_manifest(resolved)
    artifacts = load_run_artifacts(resolved, evaluation)
    ensure_artifact_dirs(artifacts)

    estimated_sequence, estimated_summary = load_estimated_pose_sequence(
        resolved.log_path,
        resolved.frame_manifest_path,
        input_type=resolved.input_type,
        absolute_start_time_seconds=evaluation.timing.absolute_start_time_seconds,
    )
    estimated_health = compute_sequence_health(estimated_sequence)

    trajectory_evaluations: dict[str, TrajectoryEvaluation] = {}
    floorplan_evaluation: FloorplanEvaluationResult | None = None
    init_pose = None
    ground_truth_sequence = None

    if evaluation.ground_truth.init_pose_csv is not None:
        init_pose = load_init_pose(
            evaluation.ground_truth.init_pose_csv,
            evaluation.ground_truth.lookup_run_name or resolved.run_name,
        )

    if evaluation.ground_truth.trajectory_txt is not None:
        ground_truth_sequence = load_tum_pose_sequence(evaluation.ground_truth.trajectory_txt)
        _, trajectory_evaluations = evaluate_trajectory_modes(
            estimated_sequence,
            ground_truth_sequence,
            init_pose=init_pose,
            ignore_initial_seconds=evaluation.timing.ignore_initial_seconds,
            association_tolerance_seconds=evaluation.timing.association_tolerance_seconds,
            alignment_modes=evaluation.trajectory.alignment_modes,
            anchor_mode=evaluation.trajectory.anchor_mode,
            init_anchor_tolerance_seconds=evaluation.trajectory.init_anchor_tolerance_seconds,
            rpe_horizon_seconds=evaluation.trajectory.rpe_horizon_seconds,
            rpe_tolerance_seconds=evaluation.trajectory.rpe_tolerance_seconds,
        )

    if evaluation.floorplan.png_path is not None and init_pose is not None:
        floorplan = load_floorplan(evaluation.floorplan.png_path, evaluation.floorplan)
        floorplan_evaluation = evaluate_floorplan_consistency(
            estimated_sequence,
            init_pose=init_pose,
            anchor_mode=evaluation.trajectory.anchor_mode,
            init_anchor_tolerance_seconds=evaluation.trajectory.init_anchor_tolerance_seconds,
            dense_log_dir=resolved.dense_log_dir,
            export_path=resolved.export_path if resolved.export_path.is_file() else None,
            floorplan=floorplan,
            config=evaluation.floorplan,
        )
    else:
        floorplan = None

    plot_paths = _collect_plot_paths(artifacts)
    if "rigid_se3" in trajectory_evaluations:
        best_fit = trajectory_evaluations["rigid_se3"]
        plot_best_fit_trajectory(best_fit, plot_paths["trajectory_best_fit"])
        plot_translation_error_timeseries(best_fit, plot_paths["translation_error_3d_vs_time"], component="3d")
        plot_translation_error_timeseries(best_fit, plot_paths["translation_error_xy_vs_time"], component="xy")
        plot_translation_error_histogram(best_fit, plot_paths["translation_error_3d_histogram"], component="3d")
        plot_translation_error_histogram(best_fit, plot_paths["translation_error_xy_histogram"], component="xy")
        plot_rpe_translation_timeseries(best_fit, plot_paths["rpe_translation_m_vs_index"])
        plot_rpe_rotation_timeseries(best_fit, plot_paths["rpe_rotation_deg_vs_index"])
        write_matched_poses_csv(
            artifacts.matched_poses_csv_path,
            timestamps=best_fit.aligned_estimated.timestamps.tolist(),
            est_positions=best_fit.aligned_estimated.positions.tolist(),
            gt_positions=best_fit.ground_truth.positions.tolist(),
            error_xy=best_fit.translation_error_xy_m.tolist(),
            error_3d=best_fit.translation_error_m.tolist(),
        )

    if floorplan is not None and floorplan_evaluation is not None:
        gt_positions = None
        if "init_anchor" in trajectory_evaluations:
            gt_positions = trajectory_evaluations["init_anchor"].ground_truth.positions
        plot_floorplan_overlay_eval_resolution(
            floorplan,
            floorplan_evaluation,
            plot_paths["floorplan_overlay"],
            ground_truth_positions=gt_positions,
        )
        plot_floorplan_overlay_eval_resolution(
            floorplan,
            floorplan_evaluation,
            plot_paths["pointcloud_floorplan_overlay"],
            ground_truth_positions=gt_positions,
        )
        plot_wall_consistency_overlay(floorplan, floorplan_evaluation, plot_paths["wall_consistency_overlay"])

    aligned_pointclouds = _write_aligned_pointclouds(
        resolved,
        artifacts,
        trajectory_evaluations,
        floorplan_evaluation,
    )

    payload: dict[str, Any] = {
        "run_name": resolved.run_name,
        "input_type": resolved.input_type,
        "resolved_config_path": str(resolved.resolved_config_path),
        "estimated_pose_summary": {
            "raw_rows": estimated_summary.raw_rows,
            "deduped_rows": estimated_summary.deduped_rows,
            "duplicate_rows": estimated_summary.duplicate_rows,
            "identical_duplicate_rows": estimated_summary.identical_duplicate_rows,
            "conflicting_duplicate_rows": estimated_summary.conflicting_duplicate_rows,
        },
        "estimated_sequence_health": {
            "pose_count": estimated_health.pose_count,
            "path_length_m": estimated_health.path_length_m,
            "xy_extent_m": estimated_health.xy_extent_m,
            "xyz_extent_m": estimated_health.xyz_extent_m,
        },
        "trajectory": {mode: evaluation_result.metrics for mode, evaluation_result in trajectory_evaluations.items()},
        "floorplan": None if floorplan_evaluation is None else floorplan_evaluation.metrics,
        "aligned_pointclouds": {
            mode: {
                "path": str(summary.output_path),
                "source_path": str(summary.source_path),
                "point_count": summary.point_count,
            }
            for mode, summary in aligned_pointclouds.items()
        },
        "pointcloud_frame_note": {
            "exported_ply": str(resolved.export_path),
            "exported_ply_frame": "VGGT reconstruction frame, suitable for raw CloudCompare inspection but not directly comparable to floorplan/map overlays.",
            "aligned_pointcloud_dir": str(artifacts.aligned_pointcloud_dir),
            "aligned_pointcloud_frame": "Evaluation map/ground-truth frame for each named alignment transform.",
        },
    }

    write_metrics_json(artifacts.metrics_json_path, payload)
    write_metrics_csv(artifacts.metrics_csv_path, payload)

    best_fit_metrics = trajectory_evaluations.get("rigid_se3")
    key_metrics: dict[str, Any] = {
        "Estimated poses (deduped)": estimated_summary.deduped_rows,
        "Estimated path length [m]": f"{estimated_health.path_length_m:.3f}",
        "Estimated XY extent [m]": f"{estimated_health.xy_extent_m:.3f}",
    }
    caveats: list[str] = []
    if best_fit_metrics is not None:
        key_metrics["ATE XY RMSE [m]"] = f"{best_fit_metrics.metrics['ate_xy_m_rmse']:.3f}"
        key_metrics["ATE 3D RMSE [m]"] = f"{best_fit_metrics.metrics['ate_3d_m_rmse']:.3f}"
        key_metrics["RPE translation RMSE [m]"] = f"{best_fit_metrics.metrics['rpe_translation_m_rmse']:.3f}"
    else:
        caveats.append("Full trajectory-vs-ground-truth evaluation was skipped because no GT trajectory txt was configured.")

    if floorplan_evaluation is not None:
        key_metrics["Wall precision"] = _format_metric(floorplan_evaluation.metrics["wall_precision"])
        key_metrics["Wall recall"] = _format_metric(floorplan_evaluation.metrics["wall_recall"])
        key_metrics["Wall distance p95 [m]"] = _format_metric(floorplan_evaluation.metrics["wall_distance_p95_m"])
        key_metrics["Point evidence source"] = floorplan_evaluation.summary.get("point_evidence_source", "n/a")
        key_metrics["Point evidence cells"] = floorplan_evaluation.summary.get("point_evidence_cells", "n/a")
        key_metrics["Point evidence height band [m]"] = f"{evaluation.floorplan.z_min_m:.2f} to {evaluation.floorplan.z_max_m:.2f}"
        key_metrics["Point evidence min points/cell"] = evaluation.floorplan.min_points_per_cell
        key_metrics["Point evidence min vertical extent [m]"] = f"{evaluation.floorplan.vertical_extent_min_m:.2f}"
    else:
        caveats.append("Floor-plan consistency evaluation was skipped because floorplan/init-pose inputs were incomplete.")

    if evaluation.ground_truth.trajectory_txt is None:
        caveats.append("This run has no configured full GT trajectory, so supervisor-facing trajectory accuracy metrics are unavailable.")

    if floorplan_evaluation is not None:
        caveats.append("Floor-plan consistency uses the as-planned PNG floorplan, not the authoritative as-built geometry.")
        caveats.append(
            "The CloudCompare export PLY is in VGGT's reconstruction frame; floorplan overlays and aligned_*.ply artifacts are transformed into the evaluation map frame."
        )
        if int(floorplan_evaluation.summary.get("point_evidence_cells", 0)) == 0:
            caveats.append("No usable wall evidence cells were extracted from the reconstruction after map anchoring and height filtering.")
    if resolved.input_type == "mp4":
        caveats.append(
            "MP4 timestamps are derived as absolute_start_time_seconds + frame_manifest timestamp_seconds; this assumes the equirect video starts at camera time 10000 s."
        )

    plots_for_report = {
        label: str(path.relative_to(artifacts.output_dir))
        for label, path in plot_paths.items()
        if path.is_file()
    }
    write_markdown_report(
        artifacts.report_path,
        title=f"Evaluation Report: {resolved.run_name}",
        metadata={
            "run_name": resolved.run_name,
            "input_type": resolved.input_type,
            "estimated_poses_txt": resolved.log_path,
            "frame_manifest": resolved.frame_manifest_path,
            "original_pointcloud_vggt_frame": resolved.export_path,
            "aligned_pointcloud_dir": artifacts.aligned_pointcloud_dir,
            "ground_truth": evaluation.ground_truth.trajectory_txt or "not configured",
            "floorplan": evaluation.floorplan.png_path or "not configured",
        },
        key_metrics=key_metrics,
        plots=plots_for_report,
        caveats=caveats,
    )

    return EvaluationResult(
        artifacts=artifacts,
        estimated_summary=estimated_summary,
        estimated_health=estimated_health,
        trajectory_evaluations=trajectory_evaluations,
        floorplan_evaluation=floorplan_evaluation,
        payload=payload,
    )
