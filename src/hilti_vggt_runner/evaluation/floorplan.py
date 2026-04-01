from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from ..export import iter_frame_logs
from .align import InitAnchorResult, apply_similarity_to_points, apply_similarity_to_pose_sequence, build_init_anchor_transform
from .config import FloorplanConfig
from .poses import InitPose, PoseSequence, load_estimated_pose_sequence, pose_sequence_to_matrices


@dataclass(frozen=True)
class FloorplanData:
    image: np.ndarray
    wall_mask: np.ndarray
    eval_wall_mask: np.ndarray
    eval_resolution_m_per_px: float


@dataclass(frozen=True)
class FloorplanEvaluationResult:
    anchor: InitAnchorResult
    anchored_trajectory: PoseSequence
    evidence_mask: np.ndarray
    corridor_mask: np.ndarray
    metrics: dict[str, float | int | str]
    summary: dict[str, float | int | str]


def load_floorplan(path: Path, config: FloorplanConfig) -> FloorplanData:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read floorplan PNG: {path}")
    wall_mask = image == 0
    factor_float = config.eval_resolution_m_per_px / config.base_resolution_m_per_px
    factor = int(round(factor_float))
    if not np.isclose(factor_float, factor):
        raise ValueError(
            "floorplan.eval_resolution_m_per_px must be an integer multiple of floorplan.base_resolution_m_per_px"
        )
    eval_wall_mask = downsample_wall_mask(wall_mask, factor)
    return FloorplanData(
        image=image,
        wall_mask=wall_mask,
        eval_wall_mask=eval_wall_mask,
        eval_resolution_m_per_px=config.eval_resolution_m_per_px,
    )


def downsample_wall_mask(wall_mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return wall_mask.copy()

    height, width = wall_mask.shape
    pad_y = (-height) % factor
    pad_x = (-width) % factor
    padded = np.pad(wall_mask, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=False)
    reshaped = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    return reshaped.any(axis=(1, 3))


def map_xy_to_grid(points_xy: np.ndarray, shape: tuple[int, int], resolution_m_per_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = np.rint(points_xy[:, 0] / resolution_m_per_px).astype(np.int64)
    rows = shape[0] - 1 - np.rint(points_xy[:, 1] / resolution_m_per_px).astype(np.int64)
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return rows, cols, valid


def _disk_kernel(radius_px: int) -> np.ndarray:
    diameter = max(1, radius_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return kernel


def rasterize_trajectory_corridor(
    anchored_trajectory: PoseSequence,
    shape: tuple[int, int],
    *,
    resolution_m_per_px: float,
    corridor_radius_m: float,
) -> np.ndarray:
    corridor = np.zeros(shape, dtype=np.uint8)
    rows, cols, valid = map_xy_to_grid(anchored_trajectory.positions[:, :2], shape, resolution_m_per_px)
    corridor[rows[valid], cols[valid]] = 1
    radius_px = max(1, int(round(corridor_radius_m / resolution_m_per_px)))
    return cv2.dilate(corridor, _disk_kernel(radius_px)).astype(bool)


def _load_points_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    pointcloud = data["pointcloud"]
    mask = data["mask"].astype(bool)
    points = pointcloud[mask]
    finite = np.isfinite(points).all(axis=1)
    return points[finite].astype(np.float64)


def _accumulate_points_into_grid(
    points_world: np.ndarray,
    *,
    transform,
    shape: tuple[int, int],
    resolution_m_per_px: float,
    z_min_m: float,
    z_max_m: float,
    counts: np.ndarray,
    zmins: np.ndarray,
    zmaxs: np.ndarray,
) -> int:
    transformed = apply_similarity_to_points(points_world, transform)
    z_mask = (transformed[:, 2] >= z_min_m) & (transformed[:, 2] <= z_max_m)
    transformed = transformed[z_mask]
    if transformed.size == 0:
        return 0
    rows, cols, valid = map_xy_to_grid(transformed[:, :2], shape, resolution_m_per_px)
    transformed = transformed[valid]
    rows = rows[valid]
    cols = cols[valid]
    if transformed.size == 0:
        return 0
    np.add.at(counts, (rows, cols), 1)
    np.minimum.at(zmins, (rows, cols), transformed[:, 2])
    np.maximum.at(zmaxs, (rows, cols), transformed[:, 2])
    return int(transformed.shape[0])


def build_point_evidence_mask(
    *,
    dense_log_dir: Path | None,
    export_path: Path | None,
    transform,
    floorplan: FloorplanData,
    config: FloorplanConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    shape = floorplan.eval_wall_mask.shape
    counts = np.zeros(shape, dtype=np.int32)
    zmins = np.full(shape, np.inf, dtype=np.float32)
    zmaxs = np.full(shape, -np.inf, dtype=np.float32)
    source_name = "raw_logs"
    point_count = 0

    if dense_log_dir is not None and dense_log_dir.is_dir() and any(dense_log_dir.glob("*.npz")) and config.prefer_raw_logs:
        for npz_path in iter_frame_logs(dense_log_dir):
            point_count += _accumulate_points_into_grid(
                _load_points_from_npz(npz_path),
                transform=transform,
                shape=shape,
                resolution_m_per_px=floorplan.eval_resolution_m_per_px,
                z_min_m=config.z_min_m,
                z_max_m=config.z_max_m,
                counts=counts,
                zmins=zmins,
                zmaxs=zmaxs,
            )
    elif export_path is not None and export_path.is_file():
        source_name = "exported_ply"
        pcd = o3d.io.read_point_cloud(str(export_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        point_count += _accumulate_points_into_grid(
            points,
            transform=transform,
            shape=shape,
            resolution_m_per_px=floorplan.eval_resolution_m_per_px,
            z_min_m=config.z_min_m,
            z_max_m=config.z_max_m,
            counts=counts,
            zmins=zmins,
            zmaxs=zmaxs,
        )
    else:
        raise RuntimeError("Neither raw frame logs nor exported point cloud are available for floorplan evaluation")

    vertical_extent = np.where(np.isfinite(zmins) & np.isfinite(zmaxs), zmaxs - zmins, 0.0)
    evidence = (counts >= config.min_points_per_cell) & (vertical_extent >= config.vertical_extent_min_m)
    return evidence, {
        "point_evidence_source": source_name,
        "point_evidence_input_points": int(point_count),
        "point_evidence_cells": int(evidence.sum()),
    }


def _distance_to_wall_m(wall_mask: np.ndarray, resolution_m_per_px: float) -> np.ndarray:
    free_mask = (~wall_mask).astype(np.uint8)
    return cv2.distanceTransform(free_mask, cv2.DIST_L2, 3) * resolution_m_per_px


def evaluate_floorplan_consistency(
    estimated_sequence: PoseSequence,
    *,
    init_pose: InitPose,
    anchor_mode: str,
    init_anchor_tolerance_seconds: float,
    dense_log_dir: Path | None,
    export_path: Path | None,
    floorplan: FloorplanData,
    config: FloorplanConfig,
) -> FloorplanEvaluationResult:
    anchor = build_init_anchor_transform(
        estimated_sequence,
        init_pose,
        mode=anchor_mode,
        max_timestamp_delta_seconds=init_anchor_tolerance_seconds,
    )
    anchored_trajectory = apply_similarity_to_pose_sequence(estimated_sequence, anchor.transform)
    corridor = rasterize_trajectory_corridor(
        anchored_trajectory,
        floorplan.eval_wall_mask.shape,
        resolution_m_per_px=floorplan.eval_resolution_m_per_px,
        corridor_radius_m=config.trajectory_corridor_m,
    )
    evidence, evidence_summary = build_point_evidence_mask(
        dense_log_dir=dense_log_dir,
        export_path=export_path,
        transform=anchor.transform,
        floorplan=floorplan,
        config=config,
    )

    wall_match_radius_px = max(1, int(round(config.wall_match_radius_m / floorplan.eval_resolution_m_per_px)))
    dilated_walls = cv2.dilate(floorplan.eval_wall_mask.astype(np.uint8), _disk_kernel(wall_match_radius_px)).astype(bool)
    dilated_evidence = cv2.dilate(evidence.astype(np.uint8), _disk_kernel(wall_match_radius_px)).astype(bool)

    evidence_roi = evidence & corridor
    walls_roi = floorplan.eval_wall_mask & corridor

    matched_evidence = evidence_roi & dilated_walls
    matched_walls = walls_roi & dilated_evidence

    precision = float(matched_evidence.sum() / evidence_roi.sum()) if evidence_roi.any() else 0.0
    recall = float(matched_walls.sum() / walls_roi.sum()) if walls_roi.any() else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    union = evidence_roi.sum() + walls_roi.sum() - matched_evidence.sum()
    iou = float(matched_evidence.sum() / union) if union else 0.0

    wall_distance_map = _distance_to_wall_m(floorplan.eval_wall_mask, floorplan.eval_resolution_m_per_px)
    evidence_distances = wall_distance_map[evidence_roi]
    distance_metrics = {
        "wall_distance_mean_m": float(np.mean(evidence_distances)) if evidence_distances.size else None,
        "wall_distance_median_m": float(np.median(evidence_distances)) if evidence_distances.size else None,
        "wall_distance_p95_m": float(np.percentile(evidence_distances, 95)) if evidence_distances.size else None,
    }

    metrics: dict[str, float | int | str] = {
        "alignment_mode": "init_anchor_floorplan",
        "anchor_timestamp": float(anchor.anchor_timestamp),
        "anchor_timestamp_delta_seconds": float(anchor.timestamp_delta_seconds),
        "anchor_frame_id": -1 if anchor.anchor_frame_id is None else anchor.anchor_frame_id,
        "trajectory_corridor_cells": int(corridor.sum()),
        "evidence_cells_in_corridor": int(evidence_roi.sum()),
        "wall_cells_in_corridor": int(walls_roi.sum()),
        "wall_precision": precision,
        "wall_recall": recall,
        "wall_f1": f1,
        "wall_iou": iou,
        **distance_metrics,
        **evidence_summary,
    }
    return FloorplanEvaluationResult(
        anchor=anchor,
        anchored_trajectory=anchored_trajectory,
        evidence_mask=evidence,
        corridor_mask=corridor,
        metrics=metrics,
        summary=evidence_summary,
    )
