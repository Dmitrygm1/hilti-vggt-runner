from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Any

import cv2
import yaml

from .config import (
    Mp4InputConfig,
    RosbagInputConfig,
    RunnerContext,
    ViewsConfig,
    derive_view_cache_root,
    ensure_layout_dirs,
    requested_physical_frame_limit,
    write_resolved_config,
)
from .rosbag import RosbagStitchConfig, extract_rosbag_frames, maybe_rotate_image
from .views import (
    FrameManifestRecord,
    RenderSummary,
    build_preview_contact_sheet,
    materialize_view_major_sequence,
    read_frame_manifest,
    render_views_from_equirect,
    select_records_by_physical_limit,
    write_frame_manifest,
)


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
    view_mode: str
    physical_frame_count: int
    extracted_frame_count: int
    stride: int
    image_folder: Path
    view_count: int
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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _physical_limit(context: RunnerContext) -> int:
    return requested_physical_frame_limit(context.sequence, context.profile)


def _existing_cached_sequence_matches(metadata_path: Path, frames_dir: Path) -> tuple[bool, dict[str, Any] | None, list[Path]]:
    frames = list_image_files(frames_dir)
    if not frames:
        return False, None, frames
    metadata = _load_yaml_if_exists(metadata_path)
    if metadata is None:
        return False, None, frames
    if int(metadata.get("extracted_frames", -1)) != len(frames):
        return False, metadata, frames
    return True, metadata, frames


def _write_view_metadata(
    metadata_path: Path,
    *,
    context: RunnerContext,
    summary: RenderSummary,
) -> None:
    _write_yaml(
        metadata_path,
        {
            "source_type": context.sequence.input.type,
            "view_mode": context.sequence.views.mode,
            "ordering": summary.ordering,
            "physical_frames": summary.physical_frame_count,
            "extracted_frames": summary.emitted_frame_count,
            "output_width": summary.output_width,
            "output_height": summary.output_height,
            "view_count": summary.view_count,
            "is_complete": _physical_limit(context) == 0,
            "requested_physical_frame_limit": None if _physical_limit(context) == 0 else _physical_limit(context),
            "preview_path": str(context.layout.preview_path),
            "frames_dir": str(context.layout.frames_dir),
        },
    )


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

    existing_ok, _, existing_frames = _existing_cached_sequence_matches(
        context.layout.source_metadata_path,
        context.layout.source_frames_dir,
    )
    if existing_ok and not force:
        return PreparationSummary(
            source_type="mp4",
            view_mode="equirect",
            physical_frame_count=len(existing_frames),
            extracted_frame_count=len(existing_frames),
            stride=stride,
            image_folder=context.layout.image_folder,
            view_count=1,
            metadata=metadata,
            actual_sample_fps=actual_sample_fps,
            output_width=metadata.width,
            output_height=metadata.height,
        )

    _reset_directory(context.layout.source_frames_dir)

    capture = cv2.VideoCapture(str(input_cfg.source_mp4))
    records: list[FrameManifestRecord] = []
    extracted_count = 0
    source_frame_index = 0
    jpeg_quality = context.sequence.extraction.jpeg_quality
    limit = 0 if frame_limit is None else frame_limit

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
                output_path = context.layout.source_frames_dir / output_name
                if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise RuntimeError(f"Failed to write extracted frame: {output_path}")
                timestamp_seconds = round(source_frame_index / metadata.source_fps, 6)
                records.append(
                    FrameManifestRecord(
                        frame_index=extracted_count,
                        physical_frame_index=extracted_count,
                        view_index=0,
                        output_name=output_name,
                        source_frame_index=source_frame_index,
                        timestamp_seconds=timestamp_seconds,
                        is_eval_primary=True,
                    )
                )
                if limit > 0 and extracted_count >= limit:
                    break
            source_frame_index += 1
    finally:
        capture.release()

    write_frame_manifest(context.layout.source_frame_manifest_path, records)
    frame_paths = list_image_files(context.layout.source_frames_dir)
    build_preview_contact_sheet(frame_paths, context.layout.source_preview_path)
    _write_yaml(
        context.layout.source_metadata_path,
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
            "requested_physical_frame_limit": None if limit == 0 else limit,
            "is_complete": limit == 0,
            "extracted_frames": extracted_count,
            "preview_path": str(context.layout.source_preview_path),
            "frames_dir": str(context.layout.source_frames_dir),
        },
    )
    if context.layout.stitch_summary_path.exists():
        context.layout.stitch_summary_path.unlink()

    return PreparationSummary(
        source_type="mp4",
        view_mode="equirect",
        physical_frame_count=extracted_count,
        extracted_frame_count=extracted_count,
        stride=stride,
        image_folder=context.layout.image_folder,
        view_count=1,
        metadata=metadata,
        actual_sample_fps=actual_sample_fps,
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

    existing_ok, _, existing_frames = _existing_cached_sequence_matches(
        context.layout.source_metadata_path,
        context.layout.source_frames_dir,
    )
    if existing_ok and not force:
        stitch_summary = _load_yaml_if_exists(context.layout.stitch_summary_path) or {}
        return PreparationSummary(
            source_type="rosbag",
            view_mode="equirect",
            physical_frame_count=len(existing_frames),
            extracted_frame_count=len(existing_frames),
            stride=input_cfg.stride,
            image_folder=context.layout.image_folder,
            view_count=1,
            output_width=int(stitch_summary.get("output_width", 0) or 0),
            output_height=int(stitch_summary.get("output_height", 0) or 0),
            paired_messages=int(stitch_summary.get("paired_messages", 0) or 0),
            sampled_pairs=int(stitch_summary.get("sampled_pairs", len(existing_frames)) or len(existing_frames)),
        )

    limit = input_cfg.max_frames if frame_limit is None else frame_limit
    summary = extract_rosbag_frames(
        RosbagStitchConfig(
            bag_path=input_cfg.rosbag_db3,
            out_dir=context.layout.source_frames_dir,
            frame_manifest_path=context.layout.source_frame_manifest_path,
            stitch_summary_path=context.layout.stitch_summary_path,
            preview_path=context.layout.source_preview_path,
            calibration_yaml=input_cfg.calibration_yaml,
            mask0=input_cfg.mask0,
            mask1=input_cfg.mask1,
            jpeg_quality=context.sequence.extraction.jpeg_quality,
            sphere_m=input_cfg.sphere_m,
            stride=input_cfg.stride,
            max_frames=limit,
            rotate_180=input_cfg.rotate_180,
            sync_tolerance_ns=input_cfg.sync_tolerance_ns,
            topic0=input_cfg.topic0,
            topic1=input_cfg.topic1,
        )
    )

    _write_yaml(
        context.layout.source_metadata_path,
        {
            "input_type": "rosbag",
            "source_path": str(input_cfg.rosbag_db3),
            "requested_physical_frame_limit": None if limit == 0 else limit,
            "is_complete": limit == 0,
            "extracted_frames": summary.extracted_frame_count,
            "stride": input_cfg.stride,
            "rotate_180": input_cfg.rotate_180,
            "sphere_m": input_cfg.sphere_m,
            "sync_tolerance_ns": input_cfg.sync_tolerance_ns,
            "preview_path": str(context.layout.source_preview_path),
            "frames_dir": str(context.layout.source_frames_dir),
        },
    )

    return PreparationSummary(
        source_type="rosbag",
        view_mode="equirect",
        physical_frame_count=summary.extracted_frame_count,
        extracted_frame_count=summary.extracted_frame_count,
        stride=input_cfg.stride,
        image_folder=context.layout.image_folder,
        view_count=1,
        output_width=summary.output_width,
        output_height=summary.output_height,
        paired_messages=summary.paired_messages,
        sampled_pairs=summary.sampled_pairs,
    )


def _render_frame_major_views(
    *,
    context: RunnerContext,
    views: ViewsConfig,
    output_root: Path,
    force: bool,
) -> RenderSummary:
    frames_dir = output_root / "frames"
    manifest_path = output_root / "frame_manifest.csv"
    preview_path = output_root / "frame_preview.jpg"
    metadata_path = output_root / "view_metadata.yaml"

    existing_ok, metadata, existing_frames = _existing_cached_sequence_matches(metadata_path, frames_dir)
    if existing_ok and not force:
        return RenderSummary(
            physical_frame_count=int(metadata.get("physical_frames", 0) or 0),
            emitted_frame_count=len(existing_frames),
            output_width=int(metadata.get("output_width", 0) or 0),
            output_height=int(metadata.get("output_height", 0) or 0),
            view_count=int(metadata.get("view_count", views.view_count) or views.view_count),
            ordering=str(metadata.get("ordering", "frame_major")),
        )

    source_records = read_frame_manifest(context.layout.source_frame_manifest_path)
    source_records = select_records_by_physical_limit(source_records, _physical_limit(context))

    render_summary = render_views_from_equirect(
        source_dir=context.layout.source_frames_dir,
        source_records=source_records,
        output_dir=frames_dir,
        manifest_path=manifest_path,
        preview_path=preview_path,
        views=views,
        jpeg_quality=context.sequence.extraction.jpeg_quality,
        bag_path=(context.sequence.input.rosbag_db3 if isinstance(context.sequence.input, RosbagInputConfig) else None),
        calibration_yaml=(
            context.sequence.input.calibration_yaml if isinstance(context.sequence.input, RosbagInputConfig) else None
        ),
    )

    _write_yaml(
        metadata_path,
        {
            "source_type": context.sequence.input.type,
            "view_mode": views.mode,
            "ordering": render_summary.ordering,
            "physical_frames": render_summary.physical_frame_count,
            "extracted_frames": render_summary.emitted_frame_count,
            "output_width": render_summary.output_width,
            "output_height": render_summary.output_height,
            "view_count": render_summary.view_count,
            "requested_physical_frame_limit": None if _physical_limit(context) == 0 else _physical_limit(context),
            "is_complete": _physical_limit(context) == 0,
            "preview_path": str(preview_path),
            "frames_dir": str(frames_dir),
        },
    )
    return render_summary


def _prepare_rendered_views(context: RunnerContext, force: bool = False) -> RenderSummary:
    views = context.sequence.views

    if views.mode == "equirect":
        source_records = read_frame_manifest(context.layout.source_frame_manifest_path)
        source_records = select_records_by_physical_limit(source_records, _physical_limit(context))
        frame_paths = [context.layout.source_frames_dir / record.output_name for record in source_records]
        if not frame_paths:
            raise RuntimeError("No source frames available for equirect preparation")
        first_image = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise RuntimeError(f"Failed to read source frame: {frame_paths[0]}")
        out_h, out_w = first_image.shape[:2]
        build_preview_contact_sheet(frame_paths, context.layout.preview_path)
        summary = RenderSummary(
            physical_frame_count=len(source_records),
            emitted_frame_count=len(source_records),
            output_width=out_w,
            output_height=out_h,
            view_count=1,
            ordering="frame_major",
        )
        _write_view_metadata(context.layout.view_metadata_path, context=context, summary=summary)
        return summary

    if views.mode == "pinhole_level_yaw_imu" and views.ordering == "view_major":
        frame_major_root = derive_view_cache_root(
            context.paths,
            context.sequence,
            context.profile,
            ordering="frame_major",
        )
        frame_major_summary = _render_frame_major_views(
            context=context,
            views=replace(views, ordering="frame_major"),
            output_root=frame_major_root,
            force=force,
        )
        existing_ok, metadata, existing_frames = _existing_cached_sequence_matches(
            context.layout.view_metadata_path,
            context.layout.frames_dir,
        )
        if existing_ok and not force:
            return RenderSummary(
                physical_frame_count=int(metadata.get("physical_frames", 0) or 0),
                emitted_frame_count=len(existing_frames),
                output_width=int(metadata.get("output_width", frame_major_summary.output_width) or frame_major_summary.output_width),
                output_height=int(metadata.get("output_height", frame_major_summary.output_height) or frame_major_summary.output_height),
                view_count=int(metadata.get("view_count", frame_major_summary.view_count) or frame_major_summary.view_count),
                ordering=str(metadata.get("ordering", "view_major")),
            )
        summary = materialize_view_major_sequence(
            source_frames_dir=frame_major_root / "frames",
            source_manifest_path=frame_major_root / "frame_manifest.csv",
            output_dir=context.layout.frames_dir,
            manifest_path=context.layout.frame_manifest_path,
            preview_path=context.layout.preview_path,
        )
        _write_view_metadata(context.layout.view_metadata_path, context=context, summary=summary)
        return summary

    summary = _render_frame_major_views(
        context=context,
        views=views,
        output_root=context.layout.view_cache_root,
        force=force,
    )
    if context.layout.view_cache_root != derive_view_cache_root(context.paths, context.sequence, context.profile):
        raise RuntimeError("Unexpected mismatch between computed and configured view cache root")
    return summary


def create_smoke_subset(context: RunnerContext, force: bool = False) -> Path:
    records = read_frame_manifest(context.layout.frame_manifest_path)
    selected_records = select_records_by_physical_limit(records, context.sequence.extraction.smoke_frame_count)
    if not selected_records:
        raise RuntimeError(f"Smoke subset would be empty for manifest {context.layout.frame_manifest_path}")

    if context.layout.smoke_frames_dir.exists() and not force:
        existing_smoke_frames = list_image_files(context.layout.smoke_frames_dir)
        if len(existing_smoke_frames) == len(selected_records):
            return context.layout.smoke_frames_dir

    _reset_directory(context.layout.smoke_frames_dir)
    for record in selected_records:
        target_path = context.layout.smoke_frames_dir / record.output_name
        target_path.symlink_to(context.layout.frames_dir / record.output_name)
    return context.layout.smoke_frames_dir


def prepare_profile_inputs(context: RunnerContext, force: bool = False) -> PreparationSummary:
    ensure_layout_dirs(context)
    write_resolved_config(context)

    limit = _physical_limit(context)
    frame_limit = None if limit == 0 else limit

    if isinstance(context.sequence.input, Mp4InputConfig):
        source_summary = extract_video_frames(context, force=force, frame_limit=frame_limit)
    else:
        source_summary = extract_rosbag_frames_for_context(context, force=force, frame_limit=frame_limit)

    view_summary = _prepare_rendered_views(context, force=force)

    if context.profile == "smoke":
        create_smoke_subset(context, force=force)

    return PreparationSummary(
        source_type=context.sequence.input.type,
        view_mode=context.sequence.views.mode,
        physical_frame_count=view_summary.physical_frame_count,
        extracted_frame_count=view_summary.emitted_frame_count,
        stride=source_summary.stride,
        image_folder=context.layout.image_folder,
        view_count=view_summary.view_count,
        metadata=source_summary.metadata,
        actual_sample_fps=source_summary.actual_sample_fps,
        output_width=view_summary.output_width,
        output_height=view_summary.output_height,
        paired_messages=source_summary.paired_messages,
        sampled_pairs=source_summary.sampled_pairs,
    )
