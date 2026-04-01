from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import (
    InitAnchorResult,
    SimilarityTransform,
    apply_similarity_to_points,
    apply_similarity_to_pose_sequence,
    build_init_anchor_transform,
    rigid_align_points,
    rotation_angle_degrees,
    sim3_align_points,
)
from .poses import InitPose, PoseSequence, interpolate_pose_sequence, pose_sequence_to_matrices


@dataclass(frozen=True)
class SequenceHealth:
    pose_count: int
    path_length_m: float
    xy_extent_m: float
    xyz_extent_m: tuple[float, float, float]


@dataclass(frozen=True)
class AssociatedTrajectory:
    estimated: PoseSequence
    ground_truth: PoseSequence
    ignored_before_timestamp: float
    dropped_before_ignore_window: int
    dropped_outside_gt_range: int


@dataclass(frozen=True)
class TrajectoryEvaluation:
    mode: str
    aligned_estimated: PoseSequence
    ground_truth: PoseSequence
    translation_error_m: np.ndarray
    translation_error_xy_m: np.ndarray
    rotation_error_deg: np.ndarray
    rpe_translation_m: np.ndarray
    rpe_rotation_deg: np.ndarray
    metrics: dict[str, float | int | str]
    anchor: InitAnchorResult | None = None


def compute_path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def compute_sequence_health(sequence: PoseSequence) -> SequenceHealth:
    mins = sequence.positions.min(axis=0)
    maxs = sequence.positions.max(axis=0)
    extents = maxs - mins
    xy_extent = float(np.linalg.norm(extents[:2]))
    return SequenceHealth(
        pose_count=sequence.count,
        path_length_m=compute_path_length(sequence.positions),
        xy_extent_m=xy_extent,
        xyz_extent_m=(float(extents[0]), float(extents[1]), float(extents[2])),
    )


def associate_estimated_with_gt(
    estimated: PoseSequence,
    ground_truth: PoseSequence,
    *,
    ignore_initial_seconds: float,
) -> AssociatedTrajectory:
    eval_start = max(float(estimated.timestamps[0]), float(ground_truth.timestamps[0])) + ignore_initial_seconds
    keep_mask = estimated.timestamps >= eval_start
    dropped_before = int((~keep_mask).sum())
    filtered_est = estimated.subset(keep_mask)
    inside_gt_mask = (filtered_est.timestamps >= ground_truth.timestamps[0]) & (
        filtered_est.timestamps <= ground_truth.timestamps[-1]
    )
    dropped_outside = int((~inside_gt_mask).sum())
    filtered_est = filtered_est.subset(inside_gt_mask)
    if filtered_est.count < 2:
        raise RuntimeError("Not enough estimated poses remain after GT association filtering")
    interpolated_gt = interpolate_pose_sequence(ground_truth, filtered_est.timestamps)
    return AssociatedTrajectory(
        estimated=filtered_est,
        ground_truth=interpolated_gt,
        ignored_before_timestamp=eval_start,
        dropped_before_ignore_window=dropped_before,
        dropped_outside_gt_range=dropped_outside,
    )


def _translation_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _compute_absolute_rotation_error_degrees(estimated: PoseSequence, ground_truth: PoseSequence) -> np.ndarray:
    est_mats = pose_sequence_to_matrices(estimated)
    gt_mats = pose_sequence_to_matrices(ground_truth)
    errors = []
    for est_mat, gt_mat in zip(est_mats, gt_mats):
        rot_error = gt_mat[:3, :3].T @ est_mat[:3, :3]
        errors.append(rotation_angle_degrees(rot_error))
    return np.asarray(errors, dtype=np.float64)


def _compute_rpe(
    estimated: PoseSequence,
    ground_truth: PoseSequence,
    *,
    horizon_seconds: float,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    est_mats = pose_sequence_to_matrices(estimated)
    gt_mats = pose_sequence_to_matrices(ground_truth)
    timestamps = estimated.timestamps
    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    for start_index, timestamp in enumerate(timestamps[:-1]):
        target_timestamp = timestamp + horizon_seconds
        future_index = int(np.argmin(np.abs(timestamps - target_timestamp)))
        if future_index <= start_index:
            continue
        if abs(timestamps[future_index] - target_timestamp) > tolerance_seconds:
            continue

        est_rel = np.linalg.inv(est_mats[start_index]) @ est_mats[future_index]
        gt_rel = np.linalg.inv(gt_mats[start_index]) @ gt_mats[future_index]
        error = np.linalg.inv(gt_rel) @ est_rel
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        rpe_rotation.append(rotation_angle_degrees(error[:3, :3]))
    return np.asarray(rpe_translation, dtype=np.float64), np.asarray(rpe_rotation, dtype=np.float64)


def _evaluate_alignment(
    associated: AssociatedTrajectory,
    *,
    full_estimated: PoseSequence,
    mode: str,
    init_pose: InitPose | None,
    anchor_mode: str,
    init_anchor_tolerance_seconds: float,
    rpe_horizon_seconds: float,
    rpe_tolerance_seconds: float,
) -> TrajectoryEvaluation:
    estimated = associated.estimated
    ground_truth = associated.ground_truth
    anchor_result: InitAnchorResult | None = None

    if mode == "rigid_se3":
        transform = rigid_align_points(estimated.positions, ground_truth.positions)
        aligned = apply_similarity_to_pose_sequence(estimated, transform)
        metrics: dict[str, float | int | str] = {
            "alignment_mode": mode,
            "alignment_scale": float(transform.scale),
        }
    elif mode == "sim3_diagnostic":
        transform = sim3_align_points(estimated.positions, ground_truth.positions)
        aligned_positions = apply_similarity_to_points(estimated.positions, transform)
        aligned = PoseSequence(
            timestamps=estimated.timestamps,
            positions=aligned_positions,
            quaternions_xyzw=estimated.quaternions_xyzw.copy(),
            frame_ids=estimated.frame_ids,
        )
        metrics = {
            "alignment_mode": mode,
            "alignment_scale": float(transform.scale),
        }
    elif mode == "init_anchor":
        if init_pose is None:
            raise RuntimeError("init_anchor evaluation requested but no init pose is available")
        anchor_result = build_init_anchor_transform(
            full_estimated,
            init_pose,
            mode=anchor_mode,
            max_timestamp_delta_seconds=init_anchor_tolerance_seconds,
        )
        aligned = apply_similarity_to_pose_sequence(estimated, anchor_result.transform)
        metrics = {
            "alignment_mode": mode,
            "alignment_scale": 1.0,
            "anchor_frame_id": -1 if anchor_result.anchor_frame_id is None else anchor_result.anchor_frame_id,
            "anchor_timestamp": float(anchor_result.anchor_timestamp),
            "anchor_timestamp_delta_seconds": float(anchor_result.timestamp_delta_seconds),
        }
    else:
        raise ValueError(f"Unsupported trajectory alignment mode: {mode}")

    translation_error = np.linalg.norm(aligned.positions - ground_truth.positions, axis=1)
    translation_error_xy = np.linalg.norm(aligned.positions[:, :2] - ground_truth.positions[:, :2], axis=1)
    rotation_error = _compute_absolute_rotation_error_degrees(aligned, ground_truth)
    rpe_translation, rpe_rotation = _compute_rpe(
        aligned,
        ground_truth,
        horizon_seconds=rpe_horizon_seconds,
        tolerance_seconds=rpe_tolerance_seconds,
    )

    metrics.update(_translation_stats("ate_3d_m", translation_error))
    metrics.update(_translation_stats("ate_xy_m", translation_error_xy))
    metrics.update(_translation_stats("rotation_error_deg", rotation_error))
    metrics.update(_translation_stats("rpe_translation_m", rpe_translation))
    metrics.update(_translation_stats("rpe_rotation_deg", rpe_rotation))
    metrics["matched_pose_count"] = int(aligned.count)

    return TrajectoryEvaluation(
        mode=mode,
        aligned_estimated=aligned,
        ground_truth=ground_truth,
        translation_error_m=translation_error,
        translation_error_xy_m=translation_error_xy,
        rotation_error_deg=rotation_error,
        rpe_translation_m=rpe_translation,
        rpe_rotation_deg=rpe_rotation,
        metrics=metrics,
        anchor=anchor_result,
    )


def evaluate_trajectory_modes(
    estimated: PoseSequence,
    ground_truth: PoseSequence,
    *,
    init_pose: InitPose | None,
    ignore_initial_seconds: float,
    association_tolerance_seconds: float,
    alignment_modes: tuple[str, ...],
    anchor_mode: str,
    init_anchor_tolerance_seconds: float,
    rpe_horizon_seconds: float,
    rpe_tolerance_seconds: float,
) -> tuple[AssociatedTrajectory, dict[str, TrajectoryEvaluation]]:
    associated = associate_estimated_with_gt(
        estimated,
        ground_truth,
        ignore_initial_seconds=ignore_initial_seconds,
    )
    evaluations: dict[str, TrajectoryEvaluation] = {}
    for mode in alignment_modes:
        evaluations[mode] = _evaluate_alignment(
            associated,
            full_estimated=estimated,
            mode=mode,
            init_pose=init_pose,
            anchor_mode=anchor_mode,
            init_anchor_tolerance_seconds=init_anchor_tolerance_seconds,
            rpe_horizon_seconds=rpe_horizon_seconds,
            rpe_tolerance_seconds=rpe_tolerance_seconds,
        )
    return associated, evaluations
