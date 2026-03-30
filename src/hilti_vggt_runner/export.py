from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class ExportSummary:
    frame_logs: int
    raw_points: int
    output_points: int
    output_path: Path


def _numeric_stem(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected numeric .npz stem, found {path.name}") from exc


def iter_frame_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*.npz"), key=_numeric_stem)


def _load_masked_frame(npz_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(npz_path)
    pointcloud = data["pointcloud"]
    mask = data["mask"].astype(bool)

    if pointcloud.shape[:2] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match pointcloud shape {pointcloud.shape} in {npz_path}")

    points = pointcloud[mask]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    colors = None
    if "colors" in data:
        colors = data["colors"][mask][finite_mask]
        if colors.size and colors.max() > 1.0:
            colors = colors.astype(np.float64) / 255.0

    return points.astype(np.float64), colors


def export_framewise_logs_to_ply(
    log_dir: Path,
    output_path: Path,
    voxel_size: float | None = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> ExportSummary:
    log_files = iter_frame_logs(log_dir)
    if not log_files:
        raise RuntimeError(f"No .npz frame logs found in {log_dir}")

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    raw_points = 0
    has_color = True

    for npz_path in log_files:
        points, colors = _load_masked_frame(npz_path)
        if points.size == 0:
            continue
        raw_points += len(points)
        all_points.append(points)
        if colors is None:
            has_color = False
        elif has_color:
            all_colors.append(colors)

    if not all_points:
        raise RuntimeError(f"All frame logs were empty after masking/finite filtering in {log_dir}")

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors) if has_color and all_colors else None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(merged_points)
    if merged_colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)

    if voxel_size is not None and voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    if nb_neighbors > 0 and std_ratio > 0 and len(point_cloud.points) >= nb_neighbors:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), point_cloud):
        raise RuntimeError(f"Failed to write point cloud to {output_path}")

    return ExportSummary(
        frame_logs=len(log_files),
        raw_points=raw_points,
        output_points=len(point_cloud.points),
        output_path=output_path,
    )
