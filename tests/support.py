from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from hilti_vggt_runner.config import RunnerContext, load_runner_context, validate_context


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_text_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def create_test_calibration(hilti_root: Path) -> tuple[Path, Path, Path]:
    config_dir = hilti_root / "config" / "hilti_openvins"
    config_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = config_dir / "kalibr_imucam_chain.yaml"
    write_yaml(
        calibration_path,
        {
            "cam0": {
                "camera_model": "kb",
                "resolution": [32, 32],
                "intrinsics": [10.0, 10.0, 15.5, 15.5],
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
                "fov_deg": 200.0,
                "T_cam_imu": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "cam1": {
                "camera_model": "kb",
                "resolution": [32, 32],
                "intrinsics": [10.0, 10.0, 15.5, 15.5],
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
                "fov_deg": 200.0,
                "T_cam_imu": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    )

    mask0 = config_dir / "mask_cam0.png"
    mask1 = config_dir / "mask_cam1.png"
    empty_mask = np.zeros((32, 32), dtype=np.uint8)
    cv2.imwrite(str(mask0), empty_mask)
    cv2.imwrite(str(mask1), empty_mask)
    return calibration_path, mask0, mask1


def create_synthetic_rosbag(path: Path, *, frame_count: int = 4, sync_delta_ns: int = 1_000_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)")
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
        conn.execute(
            "INSERT INTO topics(id, name, type, serialization_format, offered_qos_profiles) VALUES (?, ?, ?, ?, ?)",
            (1, "/cam0/image_raw/compressed", "sensor_msgs/msg/CompressedImage", "cdr", ""),
        )
        conn.execute(
            "INSERT INTO topics(id, name, type, serialization_format, offered_qos_profiles) VALUES (?, ?, ?, ?, ?)",
            (2, "/cam1/image_raw/compressed", "sensor_msgs/msg/CompressedImage", "cdr", ""),
        )

        for index in range(frame_count):
            image0 = np.zeros((32, 32, 3), dtype=np.uint8)
            image1 = np.zeros((32, 32, 3), dtype=np.uint8)
            image0[0:8, 0:8] = (0, 0, 255)
            image0[-8:, -8:] = (0, 255, 0)
            image1[0:8, -8:] = (255, 0, 0)
            image1[-8:, 0:8] = (0, 255, 255)

            ok0, encoded0 = cv2.imencode(".jpg", image0, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok1, encoded1 = cv2.imencode(".jpg", image1, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok0 or not ok1:
                raise RuntimeError("Failed to create synthetic compressed images")

            timestamp0 = 10_000_000_000_000 + index * 33_333_333
            timestamp1 = timestamp0 + sync_delta_ns
            conn.execute(
                "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
                (1, timestamp0, sqlite3.Binary(b"prefix" + encoded0.tobytes())),
            )
            conn.execute(
                "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
                (2, timestamp1, sqlite3.Binary(b"prefix" + encoded1.tobytes())),
            )

        conn.commit()
    finally:
        conn.close()
    return path


def build_context(
    tmp_path: Path,
    monkeypatch,
    *,
    profile: str = "smoke",
    input_type: str = "mp4",
    source_video_path: Path | None = None,
    rosbag_path: Path | None = None,
) -> RunnerContext:
    home_root = tmp_path / "home"
    scratch_root = tmp_path / "scratch"
    monkeypatch.setenv("HOME", str(home_root))
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch_root))

    vggt_root = home_root / "projects" / "VGGT-SLAM"
    vggt_root.mkdir(parents=True)
    _write_text_executable(vggt_root / "main.py", "# dummy main\n")

    hilti_root = home_root / "projects" / "hilti-trimble-slam-challenge-2026"
    hilti_root.mkdir(parents=True)
    calibration_path, mask0, mask1 = create_test_calibration(hilti_root)

    venv_python = home_root / "jupyter-vggt" / "bin" / "python"
    _write_text_executable(venv_python, "#!/usr/bin/env python3\n")

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

    sequence_payload: dict[str, Any] = {
        "run_name": "floor_UG1_2025-06-18_run_1",
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
    }

    if input_type == "mp4":
        if source_video_path is None:
            raise ValueError("MP4 test context requires source_video_path")
        sequence_payload["input"] = {
            "type": "mp4",
            "source_mp4": str(source_video_path),
            "sample_fps": 1.0,
            "rotate_180": True,
        }
    elif input_type == "rosbag":
        if rosbag_path is None:
            raise ValueError("Rosbag test context requires rosbag_path")
        sequence_payload["input"] = {
            "type": "rosbag",
            "rosbag_db3": str(rosbag_path),
            "calibration_yaml": str(calibration_path),
            "mask0": str(mask0),
            "mask1": str(mask1),
            "sphere_m": 5.0,
            "stride": 1,
            "max_frames": 0,
            "rotate_180": True,
            "sync_tolerance_ns": 5_000_000,
        }
    else:
        raise ValueError(f"Unsupported input_type for tests: {input_type}")

    write_yaml(sequence_path, sequence_payload)

    context = load_runner_context(paths_path, sequence_path, profile=profile)
    validate_context(context)
    return context
