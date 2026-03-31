from __future__ import annotations

import pytest

from hilti_vggt_runner.config import load_runner_context, validate_context
from hilti_vggt_runner.run import build_vggt_command

from .support import build_context, create_synthetic_rosbag, write_yaml


def test_context_expands_envvars_and_builds_layout_for_mp4(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"placeholder")

    context = build_context(tmp_path, monkeypatch, profile="smoke", input_type="mp4", source_video_path=video_path)

    expected_root = tmp_path / "scratch" / "tester" / "hilti-vggt"
    assert context.paths.outputs_root == expected_root
    assert context.layout.run_root == expected_root / "runs" / "floor_UG1_2025-06-18_run_1"
    assert context.layout.image_folder == context.layout.smoke_frames_dir
    assert context.sequence.input.type == "mp4"


def test_context_supports_rosbag_inputs(tmp_path, monkeypatch):
    rosbag_path = create_synthetic_rosbag(tmp_path / "run_1" / "rosbag" / "rosbag.db3")
    context = build_context(tmp_path, monkeypatch, profile="full", input_type="rosbag", rosbag_path=rosbag_path)

    assert context.sequence.input.type == "rosbag"
    assert context.sequence.input.rosbag_db3 == rosbag_path
    assert context.layout.stitch_summary_path.name == "stitch_summary.yaml"


def test_validate_context_fails_for_missing_rosbag_mask(tmp_path, monkeypatch):
    rosbag_path = create_synthetic_rosbag(tmp_path / "run_1" / "rosbag" / "rosbag.db3")
    context = build_context(tmp_path, monkeypatch, profile="full", input_type="rosbag", rosbag_path=rosbag_path)

    paths_path = tmp_path / "paths.yaml"
    sequence_path = tmp_path / "sequence.yaml"
    write_yaml(
        paths_path,
        {
            "vggt_root": "${HOME}/projects/VGGT-SLAM",
            "hilti_repo_root": "${HOME}/projects/hilti-trimble-slam-challenge-2026",
            "data_root": "${HOME}/data/hilti-2026",
            "outputs_root": "${SCRATCH_ROOT}/${USER}/hilti-vggt",
            "torch_home": "${SCRATCH_ROOT}/${USER}/torch-cache",
            "venv_python": "${HOME}/jupyter-vggt/bin/python",
        },
    )
    write_yaml(
        sequence_path,
        {
            "run_name": "floor_UG1_2025-06-18_run_1",
            "input": {
                "type": "rosbag",
                "rosbag_db3": str(rosbag_path),
                "calibration_yaml": str(context.sequence.input.calibration_yaml),
                "mask0": str(context.sequence.input.mask0),
                "mask1": str(tmp_path / "missing_mask.png"),
                "sphere_m": 5.0,
                "stride": 1,
                "rotate_180": True,
                "sync_tolerance_ns": 5_000_000,
            },
            "extraction": {
                "jpeg_quality": 95,
                "smoke_frame_count": 1,
            },
            "vggt": {
                "submap_size": 16,
                "overlapping_window_size": 1,
                "max_loops": 1,
                "min_disparity": 50.0,
                "conf_threshold": 25.0,
                "lc_thres": 0.95,
            },
            "export": {
                "voxel_size": None,
                "nb_neighbors": 0,
                "std_ratio": 0.0,
            },
        },
    )

    reloaded = load_runner_context(paths_path, sequence_path, profile="full")
    with pytest.raises(ValueError, match="Mask1 not found"):
        validate_context(reloaded)


def test_build_vggt_command_includes_headless_logging(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"placeholder")
    context = build_context(tmp_path, monkeypatch, profile="full", input_type="mp4", source_video_path=video_path)

    command = build_vggt_command(context)

    assert "--headless" in command
    assert "--log_results" in command
    assert str(context.layout.image_folder) in command
    assert str(context.layout.log_path) in command
