from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation
import yaml

from hilti_vggt_runner.evaluation.align import (
    SimilarityTransform,
    apply_similarity_to_pose_sequence,
    build_init_anchor_transform,
    rigid_align_points,
)
from hilti_vggt_runner.evaluation.config import load_resolved_run_config, validate_evaluation_config
from hilti_vggt_runner.evaluation.floorplan import downsample_wall_mask, map_xy_to_grid, rasterize_trajectory_corridor
from hilti_vggt_runner.evaluation.multiview import derive_multiview_evaluation_inputs
from hilti_vggt_runner.evaluation.pointcloud import write_aligned_pointcloud
from hilti_vggt_runner.evaluation.poses import InitPose, PoseSequence, load_estimated_pose_sequence
from hilti_vggt_runner.evaluation.pipeline import run_evaluation
from hilti_vggt_runner.evaluation.config import (
    EvaluationConfig,
    FloorplanConfig,
    GroundTruthConfig,
    ResolvedRunConfig,
    TimingConfig,
    TrajectoryConfig,
)
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


def test_derive_multiview_evaluation_inputs_filters_to_one_view(tmp_path):
    profile_root = tmp_path / "run" / "full"
    dense_log_dir = profile_root / "vggt" / "poses_logs"
    dense_log_dir.mkdir(parents=True, exist_ok=True)
    export_path = profile_root / "exports" / "map.ply"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text("ply\n", encoding="utf-8")
    preview_path = profile_root / "frame_preview.jpg"
    preview_path.write_bytes(b"preview")
    source_metadata_path = tmp_path / "source_metadata.yaml"
    source_metadata_path.write_text("is_complete: true\n", encoding="utf-8")

    manifest_path = tmp_path / "frame_manifest.csv"
    _write_csv(
        manifest_path,
        [
            "frame_index",
            "physical_frame_index",
            "view_index",
            "cam0_timestamp_ns",
            "cam1_timestamp_ns",
            "sync_delta_ns",
            "output_name",
            "is_eval_primary",
        ],
        [
            {
                "frame_index": 1,
                "physical_frame_index": 1,
                "view_index": 0,
                "cam0_timestamp_ns": 10_000_000_000_000,
                "cam1_timestamp_ns": 10_000_000_001_000,
                "sync_delta_ns": 1000,
                "output_name": "frame_000001.jpg",
                "is_eval_primary": True,
            },
            {
                "frame_index": 2,
                "physical_frame_index": 1,
                "view_index": 1,
                "cam0_timestamp_ns": 10_000_000_000_000,
                "cam1_timestamp_ns": 10_000_000_001_000,
                "sync_delta_ns": 1000,
                "output_name": "frame_000002.jpg",
                "is_eval_primary": False,
            },
            {
                "frame_index": 3,
                "physical_frame_index": 2,
                "view_index": 0,
                "cam0_timestamp_ns": 10_000_100_000_000,
                "cam1_timestamp_ns": 10_000_100_001_000,
                "sync_delta_ns": 1000,
                "output_name": "frame_000003.jpg",
                "is_eval_primary": True,
            },
            {
                "frame_index": 4,
                "physical_frame_index": 2,
                "view_index": 1,
                "cam0_timestamp_ns": 10_000_100_000_000,
                "cam1_timestamp_ns": 10_000_100_001_000,
                "sync_delta_ns": 1000,
                "output_name": "frame_000004.jpg",
                "is_eval_primary": False,
            },
        ],
    )
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text(
        "\n".join(
            [
                "1 0 0 0 0 0 0 1",
                "2 0 1 0 0 0 0 1",
                "3 1 0 0 0 0 0 1",
                "4 1 1 0 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_config_path = tmp_path / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "profile": "full",
                "sequence": {
                    "run_name": "multiview_test",
                    "input": {"type": "rosbag"},
                },
                "layout": {
                    "frame_manifest_path": str(manifest_path),
                    "source_metadata_path": str(source_metadata_path),
                    "profile_root": str(profile_root),
                    "log_path": str(poses_path),
                    "dense_log_dir": str(dense_log_dir),
                    "export_path": str(export_path),
                    "preview_path": str(preview_path),
                },
            },
            handle,
            sort_keys=False,
        )

    derived = derive_multiview_evaluation_inputs(resolved_config_path)

    assert derived.selected_view_index == 0
    assert derived.selected_frame_count == 2
    assert derived.physical_frame_count == 2
    assert derived.manifest_path.is_file()
    assert derived.poses_path.is_file()

    derived_manifest = derived.manifest_path.read_text(encoding="utf-8")
    assert "frame_000001.jpg" in derived_manifest
    assert "frame_000003.jpg" in derived_manifest
    assert "frame_000002.jpg" not in derived_manifest
    assert "frame_000004.jpg" not in derived_manifest

    derived_poses = derived.poses_path.read_text(encoding="utf-8")
    assert "1 0 0 0 0 0 0 1" in derived_poses
    assert "3 1 0 0 0 0 0 1" in derived_poses
    assert "2 0 1 0 0 0 0 1" not in derived_poses
    assert "4 1 1 0 0 0 0 1" not in derived_poses

    resolved = load_resolved_run_config(derived.resolved_config_path)
    assert resolved.frame_manifest_path == derived.manifest_path
    assert resolved.log_path == derived.poses_path


def test_run_evaluation_rejects_unfiltered_multiview_manifest(tmp_path):
    profile_root = tmp_path / "run" / "full"
    dense_log_dir = profile_root / "vggt" / "poses_logs"
    dense_log_dir.mkdir(parents=True, exist_ok=True)
    export_path = profile_root / "exports" / "map.ply"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text("ply\n", encoding="utf-8")
    preview_path = profile_root / "frame_preview.jpg"
    preview_path.write_bytes(b"preview")
    source_metadata_path = tmp_path / "source_metadata.yaml"
    source_metadata_path.write_text("is_complete: true\n", encoding="utf-8")

    manifest_path = tmp_path / "frame_manifest.csv"
    _write_csv(
        manifest_path,
        [
            "frame_index",
            "physical_frame_index",
            "view_index",
            "cam0_timestamp_ns",
            "cam1_timestamp_ns",
            "sync_delta_ns",
            "output_name",
            "is_eval_primary",
        ],
        [
            {
                "frame_index": 1,
                "physical_frame_index": 1,
                "view_index": 0,
                "cam0_timestamp_ns": 10_000_000_000_000,
                "cam1_timestamp_ns": 10_000_000_001_000,
                "sync_delta_ns": 1000,
                "output_name": "frame_000001.jpg",
                "is_eval_primary": True,
            },
            {
                "frame_index": 2,
                "physical_frame_index": 1,
                "view_index": 1,
                "cam0_timestamp_ns": 10_000_000_000_000,
                "cam1_timestamp_ns": 10_000_000_001_000,
                "sync_delta_ns": 1000,
                "output_name": "frame_000002.jpg",
                "is_eval_primary": False,
            },
        ],
    )
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text("1 0 0 0 0 0 0 1\n2 0 1 0 0 0 0 1\n", encoding="utf-8")
    resolved_config_path = tmp_path / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "profile": "full",
                "sequence": {
                    "run_name": "multiview_test",
                    "input": {"type": "rosbag"},
                },
                "layout": {
                    "frame_manifest_path": str(manifest_path),
                    "source_metadata_path": str(source_metadata_path),
                    "profile_root": str(profile_root),
                    "log_path": str(poses_path),
                    "dense_log_dir": str(dense_log_dir),
                    "export_path": str(export_path),
                    "preview_path": str(preview_path),
                },
            },
            handle,
            sort_keys=False,
        )

    resolved = load_resolved_run_config(resolved_config_path)
    evaluation = EvaluationConfig(
        eval_name="multiview_guard",
        run_name=None,
        no_full_gt=True,
        ground_truth=GroundTruthConfig(
            trajectory_txt=None,
            init_pose_csv=None,
            lookup_run_name=None,
        ),
        timing=TimingConfig(
            absolute_start_time_seconds=None,
            ignore_initial_seconds=0.0,
            association_tolerance_seconds=0.05,
        ),
        trajectory=TrajectoryConfig(
            alignment_modes=("init_anchor",),
            anchor_mode="init_yaw_translation",
            init_anchor_tolerance_seconds=1.0,
            rpe_horizon_seconds=1.0,
            rpe_tolerance_seconds=0.25,
        ),
        floorplan=FloorplanConfig(
            png_path=None,
            base_resolution_m_per_px=0.01,
            eval_resolution_m_per_px=0.05,
            wall_dilation_m=0.15,
            trajectory_corridor_m=3.0,
            wall_match_radius_m=0.15,
            z_min_m=0.2,
            z_max_m=2.5,
            min_points_per_cell=3,
            vertical_extent_min_m=0.5,
            prefer_raw_logs=True,
        ),
    )

    try:
        run_evaluation(resolved, evaluation)
    except RuntimeError as exc:
        assert "multiview manifest" in str(exc)
        assert "prepare_multiview_eval_inputs.py" in str(exc)
    else:
        raise AssertionError("Expected run_evaluation to reject an unfiltered multiview manifest")


def test_write_aligned_pointcloud_applies_evaluation_transform(tmp_path):
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "aligned" / "aligned_rigid_se3.ply"
    points = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 1.5]], dtype=np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64))
    assert o3d.io.write_point_cloud(str(source_path), cloud)

    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    transform = SimilarityTransform(rotation=rotation, translation=np.asarray([1.0, 2.0, 3.0]), scale=2.0)
    summary = write_aligned_pointcloud(
        mode="rigid_se3",
        source_path=source_path,
        output_path=output_path,
        transform=transform,
    )

    aligned = o3d.io.read_point_cloud(str(output_path))
    expected = 2.0 * (rotation @ points.T).T + np.asarray([1.0, 2.0, 3.0])
    assert summary.mode == "rigid_se3"
    assert summary.point_count == 2
    assert np.allclose(np.asarray(aligned.points), expected)
    assert np.allclose(np.asarray(aligned.colors), np.asarray(cloud.colors))


def test_validate_evaluation_config_requires_explicit_no_full_gt(tmp_path):
    log_path = tmp_path / "poses.txt"
    log_path.write_text("1 0 0 0 0 0 0 1\n", encoding="utf-8")
    manifest_path = tmp_path / "frame_manifest.csv"
    _write_csv(
        manifest_path,
        ["frame_index", "cam0_timestamp_ns", "cam1_timestamp_ns", "sync_delta_ns", "output_name"],
        [
            {
                "frame_index": 1,
                "cam0_timestamp_ns": 10_000_000_000_000,
                "cam1_timestamp_ns": 10_000_000_000_000,
                "sync_delta_ns": 0,
                "output_name": "frame_000001.jpg",
            },
        ],
    )
    resolved = ResolvedRunConfig(
        resolved_config_path=tmp_path / "resolved_config.yaml",
        run_name="floor_UG1_2025-06-18_run_1",
        input_type="rosbag",
        profile="full",
        frame_manifest_path=manifest_path,
        source_metadata_path=tmp_path / "source_metadata.yaml",
        profile_root=tmp_path / "profile",
        log_path=log_path,
        dense_log_dir=tmp_path / "logs",
        export_path=tmp_path / "exports" / "map.ply",
        preview_path=tmp_path / "preview.jpg",
    )
    evaluation = EvaluationConfig(
        eval_name="must_be_explicit",
        run_name=None,
        no_full_gt=False,
        ground_truth=GroundTruthConfig(
            trajectory_txt=None,
            init_pose_csv=None,
            lookup_run_name=None,
        ),
        timing=TimingConfig(
            absolute_start_time_seconds=None,
            ignore_initial_seconds=0.0,
            association_tolerance_seconds=0.05,
        ),
        trajectory=TrajectoryConfig(
            alignment_modes=("init_anchor",),
            anchor_mode="init_yaw_translation",
            init_anchor_tolerance_seconds=1.0,
            rpe_horizon_seconds=1.0,
            rpe_tolerance_seconds=0.25,
        ),
        floorplan=FloorplanConfig(
            png_path=None,
            base_resolution_m_per_px=0.01,
            eval_resolution_m_per_px=0.05,
            wall_dilation_m=0.15,
            trajectory_corridor_m=3.0,
            wall_match_radius_m=0.15,
            z_min_m=0.2,
            z_max_m=2.5,
            min_points_per_cell=3,
            vertical_extent_min_m=0.5,
            prefer_raw_logs=True,
        ),
    )

    with pytest.raises(ValueError, match="no ground_truth.trajectory_txt"):
        validate_evaluation_config(resolved, evaluation)


def test_run_evaluation_writes_aligned_clouds_and_floorplan_evidence_artifacts(tmp_path):
    profile_root = tmp_path / "run" / "full"
    dense_log_dir = profile_root / "vggt" / "poses_logs"
    dense_log_dir.mkdir(parents=True, exist_ok=True)
    export_path = profile_root / "exports" / "map.ply"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = profile_root / "frame_preview.jpg"
    preview_path.write_bytes(b"preview")
    source_metadata_path = tmp_path / "source_metadata.yaml"
    source_metadata_path.write_text("is_complete: true\n", encoding="utf-8")

    manifest_path = tmp_path / "frame_manifest.csv"
    timestamps_ns = [10_000_000_000_000, 10_001_000_000_000, 10_002_000_000_000, 10_003_000_000_000]
    _write_csv(
        manifest_path,
        ["frame_index", "cam0_timestamp_ns", "cam1_timestamp_ns", "sync_delta_ns", "output_name"],
        [
            {
                "frame_index": index,
                "cam0_timestamp_ns": timestamp_ns,
                "cam1_timestamp_ns": timestamp_ns,
                "sync_delta_ns": 0,
                "output_name": f"frame_{index:06d}.jpg",
            }
            for index, timestamp_ns in enumerate(timestamps_ns, start=1)
        ],
    )
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text(
        "\n".join(
            [
                "1 0 0 0 0 0 0 1",
                "2 1 0 0 0 0 0 1",
                "3 2 1 0 0 0 0 1",
                "4 3 1 0 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "\n".join(
            [
                "10000.000000000 1 1 0 0 0 0 1",
                "10001.000000000 2 1 0 0 0 0 1",
                "10002.000000000 3 2 0 0 0 0 1",
                "10003.000000000 4 2 0 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    init_pose_csv = tmp_path / "init_gt_poses.csv"
    init_pose_csv.write_text(
        "floor_1_2025-05-05_run_1,floor_1.png,10000.000000000,1,1,0,0,0,0,1\n",
        encoding="utf-8",
    )
    floorplan_png = tmp_path / "floor_1.png"
    floorplan_image = np.full((500, 500), 255, dtype=np.uint8)
    floorplan_image[250:254, 10:490] = 0
    cv2.imwrite(str(floorplan_png), floorplan_image)

    ply_points = np.asarray(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.2],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 1.2],
            [2.0, 1.0, 0.5],
            [2.0, 1.0, 1.2],
        ],
        dtype=np.float64,
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(ply_points)
    assert o3d.io.write_point_cloud(str(export_path), cloud)

    resolved_config_path = tmp_path / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "profile": "full",
                "sequence": {
                    "run_name": "floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300",
                    "input": {"type": "rosbag"},
                },
                "layout": {
                    "frame_manifest_path": str(manifest_path),
                    "source_metadata_path": str(source_metadata_path),
                    "profile_root": str(profile_root),
                    "log_path": str(poses_path),
                    "dense_log_dir": str(dense_log_dir),
                    "export_path": str(export_path),
                    "preview_path": str(preview_path),
                },
            },
            handle,
            sort_keys=False,
        )
    resolved = load_resolved_run_config(resolved_config_path)
    evaluation = EvaluationConfig(
        eval_name="floor1_regression",
        run_name=None,
        no_full_gt=False,
        ground_truth=GroundTruthConfig(
            trajectory_txt=gt_path,
            init_pose_csv=init_pose_csv,
            lookup_run_name="floor_1_2025-05-05_run_1",
        ),
        timing=TimingConfig(
            absolute_start_time_seconds=None,
            ignore_initial_seconds=0.0,
            association_tolerance_seconds=0.05,
        ),
        trajectory=TrajectoryConfig(
            alignment_modes=("rigid_se3", "init_anchor", "sim3_diagnostic"),
            anchor_mode="init_yaw_translation",
            init_anchor_tolerance_seconds=0.1,
            rpe_horizon_seconds=1.0,
            rpe_tolerance_seconds=0.25,
        ),
        floorplan=FloorplanConfig(
            png_path=floorplan_png,
            base_resolution_m_per_px=0.01,
            eval_resolution_m_per_px=0.05,
            wall_dilation_m=0.15,
            trajectory_corridor_m=3.0,
            wall_match_radius_m=0.15,
            z_min_m=0.2,
            z_max_m=2.5,
            min_points_per_cell=1,
            vertical_extent_min_m=0.1,
            prefer_raw_logs=False,
        ),
    )

    result = run_evaluation(resolved, evaluation)
    plots = result.artifacts.plots_dir
    aligned = result.artifacts.aligned_pointcloud_dir
    report = result.artifacts.report_path.read_text(encoding="utf-8")

    assert (plots / "trajectory_xy_best_fit.png").is_file()
    assert (plots / "pointcloud_floorplan_overlay.png").is_file()
    assert (aligned / "aligned_rigid_se3.ply").is_file()
    assert (aligned / "aligned_init_anchor.ply").is_file()
    assert (aligned / "aligned_sim3_diagnostic.ply").is_file()
    assert result.payload["aligned_pointclouds"]["init_anchor"]["path"].endswith("aligned_init_anchor.ply")
    assert result.payload["floorplan"]["point_evidence_source"] == "exported_ply"
    assert "original_pointcloud_vggt_frame" in report
    assert "CloudCompare export PLY is in VGGT's reconstruction frame" in report
