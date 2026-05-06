from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .align import SimilarityTransform, apply_similarity_to_points


@dataclass(frozen=True)
class AlignedPointcloudSummary:
    mode: str
    source_path: Path
    output_path: Path
    point_count: int


def write_aligned_pointcloud(
    *,
    mode: str,
    source_path: Path,
    output_path: Path,
    transform: SimilarityTransform,
) -> AlignedPointcloudSummary:
    """Write a PLY transformed into the same frame used by an evaluation mode."""
    if not source_path.is_file():
        raise RuntimeError(f"Source point cloud not found: {source_path}")

    source_cloud = o3d.io.read_point_cloud(str(source_path))
    points = np.asarray(source_cloud.points, dtype=np.float64)
    if points.size == 0:
        raise RuntimeError(f"Source point cloud has no points: {source_path}")

    aligned_cloud = o3d.geometry.PointCloud()
    aligned_cloud.points = o3d.utility.Vector3dVector(apply_similarity_to_points(points, transform))
    if source_cloud.has_colors():
        aligned_cloud.colors = source_cloud.colors
    if source_cloud.has_normals():
        normals = np.asarray(source_cloud.normals, dtype=np.float64)
        aligned_cloud.normals = o3d.utility.Vector3dVector((transform.rotation @ normals.T).T)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), aligned_cloud):
        raise RuntimeError(f"Failed to write aligned point cloud: {output_path}")

    return AlignedPointcloudSummary(
        mode=mode,
        source_path=source_path,
        output_path=output_path,
        point_count=int(points.shape[0]),
    )
