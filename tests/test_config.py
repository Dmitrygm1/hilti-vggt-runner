from __future__ import annotations

from pathlib import Path

from hilti_vggt_runner.run import build_vggt_command

from .support import build_context


def test_context_expands_envvars_and_builds_layout(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"placeholder")

    context = build_context(tmp_path, monkeypatch, profile="smoke", source_video_path=video_path)

    expected_root = tmp_path / "scratch" / "tester" / "hilti-vggt"
    assert context.paths.outputs_root == expected_root
    assert context.layout.run_root == expected_root / "runs" / "floor_UG1_2025-06-18_run_1"
    assert context.layout.image_folder == context.layout.smoke_frames_dir
    assert context.sequence.extraction.rotate_180 is True


def test_build_vggt_command_includes_headless_logging(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"placeholder")
    context = build_context(tmp_path, monkeypatch, profile="full", source_video_path=video_path)

    command = build_vggt_command(context)

    assert "--headless" in command
    assert "--log_results" in command
    assert str(context.layout.image_folder) in command
    assert str(context.layout.log_path) in command
