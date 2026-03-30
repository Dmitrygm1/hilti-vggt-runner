from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from hilti_vggt_runner.export import export_framewise_logs_to_ply


def test_export_framewise_logs_to_ply_preserves_masked_world_points(tmp_path):
    log_dir = tmp_path / "poses_logs"
    log_dir.mkdir()

    np.savez(
        log_dir / "1.0.npz",
        pointcloud=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [np.nan, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        mask=np.array([[True, False], [True, True]]),
        colors=np.array(
            [
                [[255, 0, 0], [0, 0, 0]],
                [[0, 255, 0], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )
    np.savez(
        log_dir / "2.0.npz",
        pointcloud=np.array([[[3.0, 3.0, 3.0]]], dtype=np.float32),
        mask=np.array([[True]]),
        colors=np.array([[[255, 255, 255]]], dtype=np.uint8),
    )

    output_path = tmp_path / "reconstruction.ply"
    summary = export_framewise_logs_to_ply(
        log_dir=log_dir,
        output_path=output_path,
        voxel_size=None,
        nb_neighbors=0,
        std_ratio=0.0,
    )

    assert summary.frame_logs == 2
    assert summary.raw_points == 3
    assert summary.output_points == 3

    point_cloud = o3d.io.read_point_cloud(str(output_path))
    assert len(point_cloud.points) == 3
    assert len(point_cloud.colors) == 3
