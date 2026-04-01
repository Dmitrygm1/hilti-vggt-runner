from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hilti_vggt_runner.evaluation.align import apply_similarity_to_pose_sequence, build_init_anchor_transform, rigid_align_points
from hilti_vggt_runner.evaluation.floorplan import downsample_wall_mask, map_xy_to_grid, rasterize_trajectory_corridor
from hilti_vggt_runner.evaluation.poses import InitPose, PoseSequence, load_estimated_pose_sequence
from hilti_vggt_runner.evaluation.trajectory import evaluate_trajectory_modes


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_estimated_pose_sequence_dedupes_duplicate_frame_ids_with_rosbag_manifest(tmp_path):
    manifest_path = tmp_path / "frame_manifest.csv"
    _write_csv(
        manifest_path,
        ["frame_index", "cam0_timestamp_ns", "cam1_timestamp_ns", "sync_delta_ns", "output_name"],
        [
            {"frame_index": 1, "cam0_timestamp_ns": 10_000_000_000_000, "cam1_timestamp_ns": 10_000_000_000_000, "sync_delta_ns": 0, "output_name": "frame_000001.jpg"},
            {"frame_index": 2, "cam0_timestamp_ns": 10_000_100_000_000, "cam1_timestamp_ns": 10_000_100_000_000, "sync_delta_ns": 0, "output_name": "frame_000002.jpg"},
            {"frame_index": 3, "cam0_timestamp_ns": 10_000_200_000_000, "cam1_timestamp_ns": 10_000_200_000_000, "sync_delta_ns": 0, "output_name": "frame_000003.jpg"},
        ],
    )
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text(
        "\n".join(
            [
                "1 0 0 0 0 0 0 1",
                "2 1 0 0 0 0 0 1",
                "2 1 0 0 0 0 0 1",
                "3 2 0 0 0 0 0 1",
                "3 3 0 0 0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )

    sequence, summary = load_estimated_pose_sequence(
        poses_path,
        manifest_path,
        input_type="rosbag",
    )

    assert summary.raw_rows == 5
    assert summary.deduped_rows == 3
    assert summary.duplicate_rows == 2
    assert summary.identical_duplicate_rows == 1
    assert summary.conflicting_duplicate_rows == 1
    assert sequence.frame_ids.tolist() == [1, 2, 3]
    assert np.allclose(sequence.timestamps, [10000.0, 10000.1, 10000.2])
    assert np.allclose(sequence.positions[-1], [3.0, 0.0, 0.0])


def test_load_estimated_pose_sequence_supports_mp4_relative_manifest_with_absolute_offset(tmp_path):
    manifest_path = tmp_path / "frame_manifest.csv"
    _write_csv(
        manifest_path,
        ["extracted_frame_index", "source_frame_index", "timestamp_seconds", "output_name"],
        [
            {"extracted_frame_index": 1, "source_frame_index": 0, "timestamp_seconds": 0.0, "output_name": "frame_000001.jpg"},
            {"extracted_frame_index": 2, "source_frame_index": 10, "timestamp_seconds": 0.333333, "output_name": "frame_000002.jpg"},
        ],
    )
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text("1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n", encoding="utf-8")

    sequence, _ = load_estimated_pose_sequence(
        poses_path,
        manifest_path,
        input_type="mp4",
        absolute_start_time_seconds=10000.0,
    )

    assert np.allclose(sequence.timestamps, [10000.0, 10000.333333])


def test_rigid_alignment_recovers_zero_ate_for_known_rigid_transform():
    timestamps = np.array([10005.0, 10006.0, 10007.0, 10008.0], dtype=np.float64)
    gt_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    gt_quats = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
    ground_truth = PoseSequence(timestamps=timestamps, positions=gt_positions, quaternions_xyzw=gt_quats)

    rotation = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    translation = np.array([10.0, -2.0, 0.5], dtype=np.float64)
    est_positions = (rotation.T @ (gt_positions - translation).T).T
    est_quats = np.tile(Rotation.from_matrix(rotation.T).as_quat(), (4, 1))
    estimated = PoseSequence(timestamps=timestamps, positions=est_positions, quaternions_xyzw=est_quats)

    _, evaluations = evaluate_trajectory_modes(
        estimated,
        ground_truth,
        init_pose=None,
        ignore_initial_seconds=0.0,
        association_tolerance_seconds=0.1,
        alignment_modes=("rigid_se3",),
        anchor_mode="init_yaw_translation",
        init_anchor_tolerance_seconds=0.5,
        rpe_horizon_seconds=1.0,
        rpe_tolerance_seconds=0.25,
    )

    rigid = evaluations["rigid_se3"]
    assert rigid.metrics["matched_pose_count"] == 4
    assert rigid.metrics["ate_xy_m_rmse"] < 1e-6
    assert rigid.metrics["ate_3d_m_rmse"] < 1e-6


def test_build_init_anchor_transform_zeros_roll_and_pitch_for_yaw_anchor():
    sequence = PoseSequence(
        timestamps=np.array([10005.0, 10006.0], dtype=np.float64),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        quaternions_xyzw=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (2, 1)),
        frame_ids=np.array([1, 2], dtype=np.int32),
    )
    init_pose = InitPose(
        run_name="run",
        floorplan_name="floor.png",
        timestamp=10005.0,
        position=np.array([5.0, 6.0, 1.2], dtype=np.float64),
        quaternion_xyzw=Rotation.from_euler("xyz", [10.0, 5.0, 90.0], degrees=True).as_quat(),
    )

    anchor = build_init_anchor_transform(
        sequence,
        init_pose,
        mode="init_yaw_translation",
        max_timestamp_delta_seconds=0.1,
    )
    anchored = apply_similarity_to_pose_sequence(sequence, anchor.transform)
    euler = Rotation.from_quat(anchored.quaternions_xyzw[0]).as_euler("xyz", degrees=True)

    assert anchor.anchor_frame_id == 1
    assert abs(anchor.timestamp_delta_seconds) < 1e-9
    assert np.allclose(anchored.positions[0], init_pose.position)
    assert abs(euler[0]) < 1e-6
    assert abs(euler[1]) < 1e-6
    assert abs(euler[2] - 90.0) < 1e-6


def test_floorplan_helpers_preserve_bottom_left_origin_and_wall_pixels():
    wall_mask = np.zeros((6, 6), dtype=bool)
    wall_mask[5, 0] = True
    wall_mask[0, 5] = True
    downsampled = downsample_wall_mask(wall_mask, factor=2)

    assert downsampled.shape == (3, 3)
    assert bool(downsampled[2, 0])
    assert bool(downsampled[0, 2])

    rows, cols, valid = map_xy_to_grid(
        np.array([[0.0, 0.0], [0.10, 0.10]], dtype=np.float64),
        (3, 3),
        resolution_m_per_px=0.05,
    )
    assert valid.tolist() == [True, True]
    assert rows.tolist()[0] == 2
    assert cols.tolist()[0] == 0

    sequence = PoseSequence(
        timestamps=np.array([0.0, 1.0], dtype=np.float64),
        positions=np.array([[0.0, 0.0, 0.0], [0.10, 0.10, 0.0]], dtype=np.float64),
        quaternions_xyzw=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (2, 1)),
    )
    corridor = rasterize_trajectory_corridor(
        sequence,
        shape=(10, 10),
        resolution_m_per_px=0.05,
        corridor_radius_m=0.10,
    )
    assert int(corridor.sum()) > 0
