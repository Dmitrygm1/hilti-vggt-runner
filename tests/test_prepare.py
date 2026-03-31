from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from hilti_vggt_runner.prepare import (
    create_smoke_subset,
    extract_rosbag_frames_for_context,
    extract_video_frames,
    list_image_files,
)
from hilti_vggt_runner.rosbag import maybe_rotate_image

from .support import build_context, create_synthetic_rosbag


def _write_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Failed to create synthetic test video")

    for _ in range(4):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[0:12, 0:12] = (0, 0, 255)
        frame[-12:, -12:] = (0, 255, 0)
        writer.write(frame)
    writer.release()


def test_extract_video_frames_rotates_and_numbers_frames(tmp_path, monkeypatch):
    video_path = tmp_path / "video.avi"
    _write_test_video(video_path)
    context = build_context(tmp_path, monkeypatch, profile="smoke", input_type="mp4", source_video_path=video_path)

    summary = extract_video_frames(context, force=True)

    extracted_frames = list_image_files(context.layout.frames_dir)
    assert summary.extracted_frame_count == 2
    assert [path.name for path in extracted_frames] == ["frame_000001.jpg", "frame_000002.jpg"]

    rotated = cv2.imread(str(extracted_frames[0]))
    assert rotated is not None
    top_left_red = float(rotated[0:12, 0:12, 2].mean())
    bottom_right_red = float(rotated[-12:, -12:, 2].mean())
    assert bottom_right_red > top_left_red
    assert context.layout.preview_path.is_file()


def test_create_smoke_subset_uses_symlinks(tmp_path, monkeypatch):
    video_path = tmp_path / "video.avi"
    _write_test_video(video_path)
    context = build_context(tmp_path, monkeypatch, profile="smoke", input_type="mp4", source_video_path=video_path)

    extract_video_frames(context, force=True, frame_limit=2)
    smoke_dir = create_smoke_subset(context, force=True)

    smoke_frames = list_image_files(smoke_dir)
    assert len(smoke_frames) == 1
    assert smoke_frames[0].is_symlink()


def test_extract_rosbag_frames_writes_manifest_preview_and_sequential_names(tmp_path, monkeypatch):
    rosbag_path = create_synthetic_rosbag(tmp_path / "run_1" / "rosbag" / "rosbag.db3", frame_count=3)
    context = build_context(tmp_path, monkeypatch, profile="full", input_type="rosbag", rosbag_path=rosbag_path)

    summary = extract_rosbag_frames_for_context(context, force=True)

    extracted_frames = list_image_files(context.layout.frames_dir)
    assert summary.extracted_frame_count == 3
    assert summary.paired_messages == 3
    assert [path.name for path in extracted_frames] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
    ]
    with context.layout.frame_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["output_name"] for row in rows] == [path.name for path in extracted_frames]
    assert all(abs(int(row["sync_delta_ns"])) <= 5_000_000 for row in rows)
    assert context.layout.frame_manifest_path.is_file()
    assert context.layout.stitch_summary_path.is_file()
    assert context.layout.preview_path.is_file()


def test_maybe_rotate_image_rotates_180():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0:2, 0:2] = (0, 0, 255)

    rotated = maybe_rotate_image(image, rotate_180=True)

    assert int(rotated[-1, -1, 2]) > 0
    assert int(rotated[0, 0, 2]) == 0
