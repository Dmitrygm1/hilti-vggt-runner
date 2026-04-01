from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .floorplan import FloorplanData, FloorplanEvaluationResult, map_xy_to_grid
from .trajectory import TrajectoryEvaluation


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_best_fit_trajectory(evaluation: TrajectoryEvaluation, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(evaluation.ground_truth.positions[:, 0], evaluation.ground_truth.positions[:, 1], label="Ground truth", color="green")
    plt.plot(
        evaluation.aligned_estimated.positions[:, 0],
        evaluation.aligned_estimated.positions[:, 1],
        label=f"Estimated ({evaluation.mode})",
        color="purple",
    )
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Trajectory XY Overlay")
    plt.legend()
    plt.axis("equal")
    _save_figure(output_path)


def plot_error_timeseries(evaluation: TrajectoryEvaluation, output_path: Path) -> None:
    relative_time = evaluation.aligned_estimated.timestamps - evaluation.aligned_estimated.timestamps[0]
    plt.figure(figsize=(8, 5))
    plt.plot(relative_time, evaluation.translation_error_xy_m, label="XY error [m]", color="tab:orange")
    plt.plot(relative_time, evaluation.translation_error_m, label="3D error [m]", color="tab:blue", alpha=0.8)
    plt.xlabel("Time since evaluation start [s]")
    plt.ylabel("Translation error [m]")
    plt.title(f"Translation Error vs Time ({evaluation.mode})")
    plt.legend()
    _save_figure(output_path)


def plot_error_histogram(evaluation: TrajectoryEvaluation, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(evaluation.translation_error_xy_m, bins=30, alpha=0.8, color="tab:orange", label="XY")
    plt.hist(evaluation.translation_error_m, bins=30, alpha=0.5, color="tab:blue", label="3D")
    plt.xlabel("Translation error [m]")
    plt.ylabel("Count")
    plt.title(f"Translation Error Histogram ({evaluation.mode})")
    plt.legend()
    _save_figure(output_path)


def plot_rpe_timeseries(evaluation: TrajectoryEvaluation, output_path: Path) -> None:
    if evaluation.rpe_translation_m.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(evaluation.rpe_translation_m, label="RPE translation [m]", color="tab:red")
    plt.plot(evaluation.rpe_rotation_deg, label="RPE rotation [deg]", color="tab:green")
    plt.xlabel("RPE pair index")
    plt.ylabel("Error")
    plt.title(f"Relative Pose Error ({evaluation.mode})")
    plt.legend()
    _save_figure(output_path)


def plot_floorplan_overlay(
    floorplan: FloorplanData,
    floorplan_result: FloorplanEvaluationResult,
    output_path: Path,
    *,
    ground_truth_positions: np.ndarray | None = None,
) -> None:
    plt.figure(figsize=(10, 7))
    plt.imshow(floorplan.image, cmap="gray", origin="upper")

    rows, cols, valid = map_xy_to_grid(
        floorplan_result.anchored_trajectory.positions[:, :2],
        floorplan.image.shape,
        floorplan.eval_resolution_m_per_px if floorplan.eval_wall_mask.shape == floorplan.image.shape else 0.01,
    )
    if valid.any():
        plt.plot(cols[valid], rows[valid], color="purple", linewidth=2, label="Estimated (init anchor)")

    if ground_truth_positions is not None and ground_truth_positions.size:
        gt_rows, gt_cols, gt_valid = map_xy_to_grid(
            ground_truth_positions[:, :2],
            floorplan.image.shape,
            floorplan.eval_resolution_m_per_px if floorplan.eval_wall_mask.shape == floorplan.image.shape else 0.01,
        )
        if gt_valid.any():
            plt.plot(gt_cols[gt_valid], gt_rows[gt_valid], color="green", linewidth=2, alpha=0.8, label="Ground truth")

    plt.title("Floorplan Overlay")
    plt.legend()
    plt.axis("off")
    _save_figure(output_path)


def plot_floorplan_overlay_eval_resolution(
    floorplan: FloorplanData,
    floorplan_result: FloorplanEvaluationResult,
    output_path: Path,
    *,
    ground_truth_positions: np.ndarray | None = None,
) -> None:
    base = np.where(floorplan.eval_wall_mask, 0, 255).astype(np.uint8)
    image = np.dstack([base, base, base])

    traj_rows, traj_cols, traj_valid = map_xy_to_grid(
        floorplan_result.anchored_trajectory.positions[:, :2],
        floorplan.eval_wall_mask.shape,
        floorplan.eval_resolution_m_per_px,
    )
    image[floorplan_result.corridor_mask] = np.clip(image[floorplan_result.corridor_mask] * 0.9, 0, 255).astype(np.uint8)
    image[floorplan_result.evidence_mask] = np.array([255, 0, 0], dtype=np.uint8)
    image[floorplan.eval_wall_mask] = np.array([0, 0, 0], dtype=np.uint8)
    if traj_valid.any():
        image[traj_rows[traj_valid], traj_cols[traj_valid]] = np.array([128, 0, 255], dtype=np.uint8)

    if ground_truth_positions is not None and ground_truth_positions.size:
        gt_rows, gt_cols, gt_valid = map_xy_to_grid(
            ground_truth_positions[:, :2],
            floorplan.eval_wall_mask.shape,
            floorplan.eval_resolution_m_per_px,
        )
        if gt_valid.any():
            image[gt_rows[gt_valid], gt_cols[gt_valid]] = np.array([0, 180, 0], dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def plot_wall_consistency_overlay(floorplan: FloorplanData, result: FloorplanEvaluationResult, output_path: Path) -> None:
    wall_mask = floorplan.eval_wall_mask
    evidence = result.evidence_mask
    corridor = result.corridor_mask
    matched = evidence & cv2.dilate(wall_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    false_positive = evidence & corridor & ~matched
    false_negative = wall_mask & corridor & ~cv2.dilate(evidence.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)

    base = np.where(wall_mask, 230, 255).astype(np.uint8)
    image = np.dstack([base, base, base])
    image[matched] = np.array([0, 170, 0], dtype=np.uint8)
    image[false_positive] = np.array([220, 40, 40], dtype=np.uint8)
    image[false_negative] = np.array([30, 90, 220], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
