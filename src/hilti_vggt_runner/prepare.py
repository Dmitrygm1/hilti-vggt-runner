from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .config import (
    Mp4InputConfig,
    RosbagInputConfig,
    RunnerContext,
    ensure_layout_dirs,
    write_resolved_config,
)
from .rosbag import RosbagStitchConfig, extract_rosbag_frames, maybe_rotate_image


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
    source_type: str
    extracted_frame_count: int
    stride: int
    image_folder: Path
    metadata: VideoMetadata | None = None
    actual_sample_fps: float | None = None
    output_width: int | None = None
    output_height: int | None = None
    paired_messages: int | None = None
    sampled_pairs: int | None = None


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
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _load_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        return None
    return loaded


def _requested_frame_limit(context: RunnerContext) -> int | None:
    if context.profile == "smoke":
        return context.sequence.extraction.smoke_frame_count
    input_cfg = context.sequence.input
    if isinstance(input_cfg, RosbagInputConfig) and input_cfg.max_frames > 0:
        return input_cfg.max_frames
    return None


def _build_cache_key(context: RunnerContext) -> dict[str, Any]:
    input_cfg = context.sequence.input
    if isinstance(input_cfg, Mp4InputConfig):
        return {
            "input_type": "mp4",
            "source_path": str(input_cfg.source_mp4),
            "sample_fps": input_cfg.sample_fps,
            "rotate_180": input_cfg.rotate_180,
            "jpeg_quality": context.sequence.extraction.jpeg_quality,
        }
    return {
        "input_type": "rosbag",
        "source_path": str(input_cfg.rosbag_db3),
        "calibration_yaml": str(input_cfg.calibration_yaml),
        "mask0": str(input_cfg.mask0) if input_cfg.mask0 else None,
        "mask1": str(input_cfg.mask1) if input_cfg.mask1 else None,
        "sphere_m": input_cfg.sphere_m,
        "stride": input_cfg.stride,
        "rotate_180": input_cfg.rotate_180,
        "sync_tolerance_ns": input_cfg.sync_tolerance_ns,
        "topic0": input_cfg.topic0,
        "topic1": input_cfg.topic1,
        "jpeg_quality": context.sequence.extraction.jpeg_quality,
    }


def _write_source_metadata(context: RunnerContext, payload: dict[str, Any]) -> None:
    context.layout.source_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with context.layout.source_metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _existing_extraction_matches(context: RunnerContext) -> tuple[bool, dict[str, Any] | None, list[Path]]:
    existing_frames = list_image_files(context.layout.frames_dir)
    if not existing_frames:
        return False, None, existing_frames

    metadata = _load_yaml_if_exists(context.layout.source_metadata_path)
    if metadata is None:
        return False, None, existing_frames
    if metadata.get("cache_key") != _build_cache_key(context):
        return False, metadata, existing_frames
    if int(metadata.get("extracted_frames", -1)) != len(existing_frames):
        return False, metadata, existing_frames

    requested_limit = _requested_frame_limit(context)
    extracted_frames = len(existing_frames)
    if requested_limit is not None and extracted_frames < requested_limit:
        return False, metadata, existing_frames
    if requested_limit is None and not bool(metadata.get("is_complete", False)):
        return False, metadata, existing_frames
    return True, metadata, existing_frames


def build_preview_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    if not frame_paths:
        return

    sample_count = min(4, len(frame_paths))
    indices = [math.floor(index) for index in list(np.linspace(0, len(frame_paths) - 1, num=sample_count))]
    images = []
    for index in indices:
        image = cv2.imread(str(frame_paths[index]))
        if image is not None:
            images.append(image)
    if not images:
        return

    tile_height = min(240, min(image.shape[0] for image in images))
    tiles = []
    for image in images:
        scale = tile_height / image.shape[0]
        tile_width = max(1, int(round(image.shape[1] * scale)))
        tiles.append(cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA))

    max_tile_width = max(tile.shape[1] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        if tile.shape[1] < max_tile_width:
            pad = np.zeros((tile.shape[0], max_tile_width - tile.shape[1], 3), dtype=tile.dtype)
            tile = np.hstack([tile, pad])
        padded_tiles.append(tile)

    if len(padded_tiles) == 1:
        sheet = padded_tiles[0]
    else:
        while len(padded_tiles) < 4:
            padded_tiles.append(np.zeros_like(padded_tiles[0]))
        top = np.hstack([padded_tiles[0], padded_tiles[1]])
        bottom = np.hstack([padded_tiles[2], padded_tiles[3]])
        sheet = np.vstack([top, bottom])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def extract_video_frames(
    context: RunnerContext,
    force: bool = False,
    frame_limit: int | None = None,
) -> PreparationSummary:
    input_cfg = context.sequence.input
    if not isinstance(input_cfg, Mp4InputConfig):
        raise TypeError("extract_video_frames only supports MP4 input contexts")

    ensure_layout_dirs(context)
    write_resolved_config(context)

    metadata = probe_video(input_cfg.source_mp4)
    stride = _compute_stride(metadata.source_fps, input_cfg.sample_fps)
    actual_sample_fps = metadata.source_fps / stride

    existing_ok, existing_metadata, existing_frames = _existing_extraction_matches(context)
    if existing_ok and not force:
        return PreparationSummary(
            source_type="mp4",
            metadata=metadata,
            extracted_frame_count=len(existing_frames),
            stride=stride,
            actual_sample_fps=actual_sample_fps,
            image_folder=context.layout.image_folder,
            output_width=metadata.width,
            output_height=metadata.height,
        )

    _reset_directory(context.layout.frames_dir)

    capture = cv2.VideoCapture(str(input_cfg.source_mp4))
    rows: list[dict[str, str | int | float]] = []
    extracted_count = 0
    source_frame_index = 0
    jpeg_quality = context.sequence.extraction.jpeg_quality

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for extraction: {input_cfg.source_mp4}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_frame_index % stride == 0:
                extracted_count += 1
                frame = maybe_rotate_image(frame, input_cfg.rotate_180)

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
                        "frame_index": extracted_count,
                        "source_frame_index": source_frame_index,
                        "timestamp_seconds": round(source_frame_index / metadata.source_fps, 6),
                        "output_name": output_name,
                    }
                )

                if frame_limit is not None and extracted_count >= frame_limit:
                    break

            source_frame_index += 1
    finally:
        capture.release()

    with context.layout.frame_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "source_frame_index", "timestamp_seconds", "output_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    frame_paths = list_image_files(context.layout.frames_dir)
    build_preview_contact_sheet(frame_paths, context.layout.preview_path)

    _write_source_metadata(
        context,
        {
            "input_type": "mp4",
            "source_path": str(input_cfg.source_mp4),
            "source_fps": metadata.source_fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "duration_seconds": metadata.duration_seconds,
            "stride": stride,
            "target_sample_fps": input_cfg.sample_fps,
            "actual_sample_fps": actual_sample_fps,
            "rotate_180": input_cfg.rotate_180,
            "requested_frame_limit": frame_limit,
            "is_complete": frame_limit is None,
            "extracted_frames": extracted_count,
            "preview_path": str(context.layout.preview_path),
            "cache_key": _build_cache_key(context),
        },
    )

    if context.layout.stitch_summary_path.exists():
        context.layout.stitch_summary_path.unlink()

    return PreparationSummary(
        source_type="mp4",
        metadata=metadata,
        extracted_frame_count=extracted_count,
        stride=stride,
        actual_sample_fps=actual_sample_fps,
        image_folder=context.layout.image_folder,
        output_width=metadata.width,
        output_height=metadata.height,
    )


def extract_rosbag_frames_for_context(
    context: RunnerContext,
    force: bool = False,
    frame_limit: int | None = None,
) -> PreparationSummary:
    input_cfg = context.sequence.input
    if not isinstance(input_cfg, RosbagInputConfig):
        raise TypeError("extract_rosbag_frames_for_context only supports rosbag input contexts")

    ensure_layout_dirs(context)
    write_resolved_config(context)

    existing_ok, _, existing_frames = _existing_extraction_matches(context)
    if existing_ok and not force:
        stitch_summary = _load_yaml_if_exists(context.layout.stitch_summary_path) or {}
        return PreparationSummary(
            source_type="rosbag",
            extracted_frame_count=len(existing_frames),
            stride=input_cfg.stride,
            image_folder=context.layout.image_folder,
            output_width=int(stitch_summary.get("output_width", 0) or 0),
            output_height=int(stitch_summary.get("output_height", 0) or 0),
            paired_messages=int(stitch_summary.get("paired_messages", 0) or 0),
            sampled_pairs=int(stitch_summary.get("sampled_pairs", len(existing_frames)) or len(existing_frames)),
        )

    max_frames = frame_limit if frame_limit is not None else input_cfg.max_frames
    summary = extract_rosbag_frames(
        RosbagStitchConfig(
            bag_path=input_cfg.rosbag_db3,
            out_dir=context.layout.frames_dir,
            frame_manifest_path=context.layout.frame_manifest_path,
            stitch_summary_path=context.layout.stitch_summary_path,
            preview_path=context.layout.preview_path,
            calibration_yaml=input_cfg.calibration_yaml,
            mask0=input_cfg.mask0,
            mask1=input_cfg.mask1,
            jpeg_quality=context.sequence.extraction.jpeg_quality,
            sphere_m=input_cfg.sphere_m,
            stride=input_cfg.stride,
            max_frames=max_frames,
            rotate_180=input_cfg.rotate_180,
            sync_tolerance_ns=input_cfg.sync_tolerance_ns,
            topic0=input_cfg.topic0,
            topic1=input_cfg.topic1,
        )
    )

    _write_source_metadata(
        context,
        {
            "input_type": "rosbag",
            "source_path": str(input_cfg.rosbag_db3),
            "requested_frame_limit": max_frames if max_frames > 0 else None,
            "is_complete": max_frames == 0,
            "extracted_frames": summary.extracted_frame_count,
            "stride": input_cfg.stride,
            "rotate_180": input_cfg.rotate_180,
            "sphere_m": input_cfg.sphere_m,
            "sync_tolerance_ns": input_cfg.sync_tolerance_ns,
            "cache_key": _build_cache_key(context),
            "preview_path": str(context.layout.preview_path),
        },
    )

    return PreparationSummary(
        source_type="rosbag",
        extracted_frame_count=summary.extracted_frame_count,
        stride=input_cfg.stride,
        image_folder=context.layout.image_folder,
        output_width=summary.output_width,
        output_height=summary.output_height,
        paired_messages=summary.paired_messages,
        sampled_pairs=summary.sampled_pairs,
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
    requested_limit = _requested_frame_limit(context)

    if isinstance(context.sequence.input, Mp4InputConfig):
        summary = extract_video_frames(context, force=force, frame_limit=requested_limit)
    else:
        summary = extract_rosbag_frames_for_context(context, force=force, frame_limit=requested_limit)

    if context.profile == "smoke":
        create_smoke_subset(context, force=force)
    return summary
