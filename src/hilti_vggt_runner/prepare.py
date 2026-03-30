from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from .config import RunnerContext, ensure_layout_dirs, write_resolved_config


@dataclass(frozen=True)
class VideoMetadata:
    source_fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.source_fps if self.source_fps else 0.0


@dataclass(frozen=True)
class PreparationSummary:
    metadata: VideoMetadata
    extracted_frame_count: int
    stride: int
    actual_sample_fps: float
    image_folder: Path


def probe_video(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        metadata = VideoMetadata(
            source_fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        capture.release()

    if metadata.source_fps <= 0 or metadata.frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}: {metadata}")
    return metadata


def _compute_stride(source_fps: float, target_fps: float) -> int:
    return max(1, int(round(source_fps / target_fps)))


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def extract_video_frames(context: RunnerContext, force: bool = False) -> PreparationSummary:
    ensure_layout_dirs(context)
    write_resolved_config(context)

    metadata = probe_video(context.sequence.source_mp4)
    stride = _compute_stride(metadata.source_fps, context.sequence.extraction.sample_fps)
    actual_sample_fps = metadata.source_fps / stride

    existing_frames = list_image_files(context.layout.frames_dir) if context.layout.frames_dir.exists() else []
    if existing_frames and context.layout.frame_manifest_path.exists() and not force:
        return PreparationSummary(
            metadata=metadata,
            extracted_frame_count=len(existing_frames),
            stride=stride,
            actual_sample_fps=actual_sample_fps,
            image_folder=context.layout.image_folder,
        )

    _reset_directory(context.layout.frames_dir)

    capture = cv2.VideoCapture(str(context.sequence.source_mp4))
    rows: list[dict[str, str | int | float]] = []
    extracted_count = 0
    source_frame_index = 0
    jpeg_quality = context.sequence.extraction.jpeg_quality

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for extraction: {context.sequence.source_mp4}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_frame_index % stride == 0:
                extracted_count += 1
                if context.sequence.extraction.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                output_name = f"frame_{extracted_count:06d}.jpg"
                output_path = context.layout.frames_dir / output_name
                success = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not success:
                    raise RuntimeError(f"Failed to write extracted frame: {output_path}")

                rows.append(
                    {
                        "extracted_frame_index": extracted_count,
                        "source_frame_index": source_frame_index,
                        "timestamp_seconds": round(source_frame_index / metadata.source_fps, 6),
                        "output_name": output_name,
                    }
                )

            source_frame_index += 1
    finally:
        capture.release()

    with context.layout.frame_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["extracted_frame_index", "source_frame_index", "timestamp_seconds", "output_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with context.layout.video_metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "source_video": str(context.sequence.source_mp4),
                "source_fps": metadata.source_fps,
                "frame_count": metadata.frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "duration_seconds": metadata.duration_seconds,
                "target_sample_fps": context.sequence.extraction.sample_fps,
                "actual_sample_fps": actual_sample_fps,
                "stride": stride,
                "rotate_180": context.sequence.extraction.rotate_180,
            },
            handle,
            sort_keys=False,
        )

    return PreparationSummary(
        metadata=metadata,
        extracted_frame_count=extracted_count,
        stride=stride,
        actual_sample_fps=actual_sample_fps,
        image_folder=context.layout.image_folder,
    )


def create_smoke_subset(context: RunnerContext, force: bool = False) -> Path:
    frame_files = list_image_files(context.layout.frames_dir)
    if not frame_files:
        raise RuntimeError(f"No extracted frames found in {context.layout.frames_dir}")

    if context.layout.smoke_frames_dir.exists() and not force:
        existing_smoke_frames = list_image_files(context.layout.smoke_frames_dir)
        if existing_smoke_frames:
            return context.layout.smoke_frames_dir

    _reset_directory(context.layout.smoke_frames_dir)

    subset_count = min(context.sequence.extraction.smoke_frame_count, len(frame_files))
    if subset_count <= 0:
        raise RuntimeError("Smoke subset would be empty")

    for frame_path in frame_files[:subset_count]:
        target_path = context.layout.smoke_frames_dir / frame_path.name
        target_path.symlink_to(frame_path)

    return context.layout.smoke_frames_dir


def prepare_profile_inputs(context: RunnerContext, force: bool = False) -> PreparationSummary:
    summary = extract_video_frames(context, force=force)
    if context.profile == "smoke":
        create_smoke_subset(context, force=force)
    return summary
