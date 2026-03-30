from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hilti_vggt_runner.prepare import create_smoke_subset, extract_video_frames, list_image_files

from .support import build_context


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
    context = build_context(tmp_path, monkeypatch, profile="smoke", source_video_path=video_path)

    summary = extract_video_frames(context, force=True)

    extracted_frames = list_image_files(context.layout.frames_dir)
    assert summary.extracted_frame_count == 2
    assert [path.name for path in extracted_frames] == ["frame_000001.jpg", "frame_000002.jpg"]

    rotated = cv2.imread(str(extracted_frames[0]))
    assert rotated is not None
    top_left_red = float(rotated[0:12, 0:12, 2].mean())
    bottom_right_red = float(rotated[-12:, -12:, 2].mean())
    assert bottom_right_red > top_left_red


def test_create_smoke_subset_uses_symlinks(tmp_path, monkeypatch):
    video_path = tmp_path / "video.avi"
    _write_test_video(video_path)
    context = build_context(tmp_path, monkeypatch, profile="smoke", source_video_path=video_path)

    extract_video_frames(context, force=True)
    smoke_dir = create_smoke_subset(context, force=True)

    smoke_frames = list_image_files(smoke_dir)
    assert len(smoke_frames) == 1
    assert smoke_frames[0].is_symlink()
