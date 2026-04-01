from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .poses import InitPose, PoseSequence, matrices_to_pose_sequence, pose_sequence_to_matrices


@dataclass(frozen=True)
class SimilarityTransform:
    rotation: np.ndarray
    translation: np.ndarray
    scale: float = 1.0

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class InitAnchorResult:
    transform: SimilarityTransform
    anchor_frame_id: int | None
    anchor_timestamp: float
    timestamp_delta_seconds: float


def rigid_align_points(est_points: np.ndarray, gt_points: np.ndarray) -> SimilarityTransform:
    if est_points.shape != gt_points.shape or est_points.shape[1] != 3:
        raise ValueError("rigid_align_points expects two arrays of shape (N, 3)")
    if est_points.shape[0] < 3:
        raise ValueError("rigid_align_points requires at least 3 points")

    est_mean = est_points.mean(axis=0)
    gt_mean = gt_points.mean(axis=0)
    est_centered = est_points - est_mean
    gt_centered = gt_points - gt_mean

    covariance = gt_centered.T @ est_centered / est_points.shape[0]
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    translation = gt_mean - rotation @ est_mean
    return SimilarityTransform(rotation=rotation, translation=translation, scale=1.0)


def sim3_align_points(est_points: np.ndarray, gt_points: np.ndarray) -> SimilarityTransform:
    if est_points.shape != gt_points.shape or est_points.shape[1] != 3:
        raise ValueError("sim3_align_points expects two arrays of shape (N, 3)")
    if est_points.shape[0] < 3:
        raise ValueError("sim3_align_points requires at least 3 points")

    est_mean = est_points.mean(axis=0)
    gt_mean = gt_points.mean(axis=0)
    est_centered = est_points - est_mean
    gt_centered = gt_points - gt_mean

    covariance = gt_centered.T @ est_centered / est_points.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    est_variance = np.mean(np.sum(est_centered**2, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / est_variance)
    translation = gt_mean - scale * (rotation @ est_mean)
    return SimilarityTransform(rotation=rotation, translation=translation, scale=scale)


def apply_similarity_to_points(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    return (transform.scale * (transform.rotation @ points.T)).T + transform.translation


def apply_similarity_to_pose_sequence(sequence: PoseSequence, transform: SimilarityTransform) -> PoseSequence:
    matrices = pose_sequence_to_matrices(sequence)
    transformed = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], sequence.count, axis=0)
    transformed[:, :3, :3] = transform.rotation[None, :, :] @ matrices[:, :3, :3]
    transformed[:, :3, 3] = apply_similarity_to_points(sequence.positions, transform)
    return matrices_to_pose_sequence(transformed, sequence.timestamps, frame_ids=sequence.frame_ids)


def rotation_angle_degrees(rotation_error_matrix: np.ndarray) -> float:
    angle = Rotation.from_matrix(rotation_error_matrix).magnitude()
    return float(np.degrees(angle))


def build_init_anchor_transform(
    sequence: PoseSequence,
    init_pose: InitPose,
    *,
    mode: str,
    max_timestamp_delta_seconds: float,
) -> InitAnchorResult:
    deltas = np.abs(sequence.timestamps - init_pose.timestamp)
    anchor_index = int(np.argmin(deltas))
    timestamp_delta = float(deltas[anchor_index])
    if timestamp_delta > max_timestamp_delta_seconds:
        raise RuntimeError(
            "Initial pose anchor could not be matched to an estimated pose within tolerance: "
            f"best_delta={timestamp_delta:.6f}s tolerance={max_timestamp_delta_seconds:.6f}s"
        )

    estimate_matrix = pose_sequence_to_matrices(sequence.subset(np.array([index == anchor_index for index in range(sequence.count)])))[0]
    init_matrix = np.eye(4, dtype=np.float64)
    init_matrix[:3, :3] = Rotation.from_quat(init_pose.quaternion_xyzw).as_matrix()
    init_matrix[:3, 3] = init_pose.position

    map_from_world = init_matrix @ np.linalg.inv(estimate_matrix)
    rotation = map_from_world[:3, :3]
    translation = map_from_world[:3, 3]

    if mode == "init_yaw_translation":
        yaw = Rotation.from_matrix(rotation).as_euler("xyz")[2]
        rotation = Rotation.from_euler("z", yaw).as_matrix()
    elif mode != "init_se3":
        raise ValueError(f"Unsupported init anchor mode: {mode}")

    return InitAnchorResult(
        transform=SimilarityTransform(rotation=rotation, translation=translation, scale=1.0),
        anchor_frame_id=None if sequence.frame_ids is None else int(sequence.frame_ids[anchor_index]),
        anchor_timestamp=float(sequence.timestamps[anchor_index]),
        timestamp_delta_seconds=timestamp_delta,
    )
