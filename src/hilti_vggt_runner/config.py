from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

ProfileName = Literal["smoke", "full"]
InputType = Literal["mp4", "rosbag"]
ViewMode = Literal["equirect", "pinhole_fixed", "pinhole_level_imu", "pinhole_level_yaw_imu"]
ViewOrdering = Literal["frame_major", "view_major"]


def _expand_envvars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_envvars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_envvars(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at {path}, found {type(loaded).__name__}")
    return _expand_envvars(loaded)


def _as_path(raw_value: str | os.PathLike[str]) -> Path:
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return path.resolve()


def _as_optional_path(raw_value: str | os.PathLike[str] | None) -> Path | None:
    if raw_value in (None, ""):
        return None
    return _as_path(raw_value)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "item"


def _short_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_to_serializable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _infer_source_label(path: Path) -> str:
    if path.name == "rosbag.db3" and len(path.parts) >= 5:
        label = "_".join(path.parts[-5:-1])
    else:
        label = path.stem or path.name
    return _slugify(label)


@dataclass(frozen=True)
class PathsConfig:
    vggt_root: Path
    hilti_repo_root: Path
    data_root: Path
    outputs_root: Path
    torch_home: Path
    venv_python: Path


@dataclass(frozen=True)
class ExtractionConfig:
    jpeg_quality: int
    smoke_frame_count: int


@dataclass(frozen=True)
class Mp4InputConfig:
    type: Literal["mp4"]
    source_mp4: Path
    sample_fps: float
    rotate_180: bool

    @property
    def source_path(self) -> Path:
        return self.source_mp4


@dataclass(frozen=True)
class RosbagInputConfig:
    type: Literal["rosbag"]
    rosbag_db3: Path
    calibration_yaml: Path
    mask0: Path | None
    mask1: Path | None
    sphere_m: float
    stride: int
    max_frames: int
    rotate_180: bool
    sync_tolerance_ns: int
    topic0: str
    topic1: str

    @property
    def source_path(self) -> Path:
        return self.rosbag_db3


InputConfig = Mp4InputConfig | RosbagInputConfig


@dataclass(frozen=True)
class ViewsConfig:
    mode: ViewMode
    width: int | None
    height: int | None
    fov_deg: float | None
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    yaws_deg: tuple[float, ...]
    ordering: ViewOrdering
    rotate_180: bool
    imu_tau: float
    time_offset_ns: int
    use_yaml_timeshift: bool
    max_physical_frames: int
    evaluation_view_index: int

    @property
    def is_multiview(self) -> bool:
        return self.mode == "pinhole_level_yaw_imu"

    @property
    def requires_imu(self) -> bool:
        return self.mode in {"pinhole_level_imu", "pinhole_level_yaw_imu"}

    @property
    def view_count(self) -> int:
        if self.mode == "pinhole_level_yaw_imu":
            return len(self.yaws_deg)
        if self.mode in {"equirect", "pinhole_fixed", "pinhole_level_imu"}:
            return 1
        raise ValueError(f"Unsupported view mode: {self.mode}")


@dataclass(frozen=True)
class VggtConfig:
    submap_size: int
    overlapping_window_size: int
    max_loops: int
    min_disparity: float
    conf_threshold: float
    lc_thres: float
    disable_flow_keyframes: bool
    vis_voxel_size: float | None = None


@dataclass(frozen=True)
class ExportConfig:
    voxel_size: float | None
    nb_neighbors: int
    std_ratio: float


@dataclass(frozen=True)
class SequenceConfig:
    run_name: str
    input: InputConfig
    views: ViewsConfig
    extraction: ExtractionConfig
    vggt: VggtConfig
    export: ExportConfig


@dataclass(frozen=True)
class LayoutConfig:
    run_root: Path
    prepared_root: Path
    source_cache_root: Path
    source_frames_dir: Path
    source_frame_manifest_path: Path
    source_metadata_path: Path
    source_preview_path: Path
    stitch_summary_path: Path
    view_cache_root: Path
    frames_dir: Path
    smoke_frames_dir: Path
    frame_manifest_path: Path
    view_metadata_path: Path
    preview_path: Path
    profile_root: Path
    image_folder: Path
    vggt_output_dir: Path
    log_path: Path
    dense_log_dir: Path
    export_dir: Path
    export_path: Path
    command_log_path: Path
    resolved_config_path: Path
    evaluation_inputs_root: Path


@dataclass(frozen=True)
class RunnerContext:
    paths: PathsConfig
    sequence: SequenceConfig
    layout: LayoutConfig
    profile: ProfileName

    def as_dict(self) -> dict[str, Any]:
        return _to_serializable(
            {
                "profile": self.profile,
                "paths": asdict(self.paths),
                "sequence": asdict(self.sequence),
                "layout": asdict(self.layout),
            }
        )


def requested_physical_frame_limit(sequence: SequenceConfig, profile: ProfileName) -> int:
    if profile == "smoke":
        return sequence.extraction.smoke_frame_count
    if sequence.views.max_physical_frames > 0:
        return sequence.views.max_physical_frames
    if isinstance(sequence.input, RosbagInputConfig) and sequence.input.max_frames > 0:
        return sequence.input.max_frames
    return 0


def _parse_input_config(seq_cfg: dict[str, Any]) -> InputConfig:
    extraction_cfg = seq_cfg.get("extraction", {})
    input_cfg = seq_cfg.get("input")

    if input_cfg is None:
        return Mp4InputConfig(
            type="mp4",
            source_mp4=_as_path(seq_cfg["source_mp4"]),
            sample_fps=float(extraction_cfg.get("sample_fps", 3.0)),
            rotate_180=bool(extraction_cfg.get("rotate_180", False)),
        )

    if not isinstance(input_cfg, dict):
        raise ValueError("Sequence 'input' must be a mapping")

    input_type = str(input_cfg.get("type", "mp4")).strip().lower()
    if input_type == "mp4":
        source_mp4 = input_cfg.get("source_mp4", seq_cfg.get("source_mp4"))
        if source_mp4 in (None, ""):
            raise ValueError("MP4 input requires input.source_mp4")
        return Mp4InputConfig(
            type="mp4",
            source_mp4=_as_path(source_mp4),
            sample_fps=float(input_cfg.get("sample_fps", extraction_cfg.get("sample_fps", 3.0))),
            rotate_180=bool(input_cfg.get("rotate_180", extraction_cfg.get("rotate_180", False))),
        )

    if input_type == "rosbag":
        rosbag_db3 = input_cfg.get("rosbag_db3")
        calibration_yaml = input_cfg.get("calibration_yaml")
        if rosbag_db3 in (None, ""):
            raise ValueError("Rosbag input requires input.rosbag_db3")
        if calibration_yaml in (None, ""):
            raise ValueError("Rosbag input requires input.calibration_yaml")
        return RosbagInputConfig(
            type="rosbag",
            rosbag_db3=_as_path(rosbag_db3),
            calibration_yaml=_as_path(calibration_yaml),
            mask0=_as_optional_path(input_cfg.get("mask0")),
            mask1=_as_optional_path(input_cfg.get("mask1")),
            sphere_m=float(input_cfg.get("sphere_m", 10.0)),
            stride=int(input_cfg.get("stride", 10)),
            max_frames=int(input_cfg.get("max_frames", 0)),
            rotate_180=bool(input_cfg.get("rotate_180", True)),
            sync_tolerance_ns=int(input_cfg.get("sync_tolerance_ns", 5_000_000)),
            topic0=str(input_cfg.get("topic0", "/cam0/image_raw/compressed")),
            topic1=str(input_cfg.get("topic1", "/cam1/image_raw/compressed")),
        )

    raise ValueError(f"Unsupported input.type: {input_type}")


def _parse_views_config(seq_cfg: dict[str, Any], input_cfg: InputConfig) -> ViewsConfig:
    views_cfg = seq_cfg.get("views") or {}
    if not isinstance(views_cfg, dict):
        raise ValueError("Sequence 'views' must be a mapping when provided")

    mode = str(views_cfg.get("mode", "equirect")).strip().lower()
    if mode not in {"equirect", "pinhole_fixed", "pinhole_level_imu", "pinhole_level_yaw_imu"}:
        raise ValueError(f"Unsupported views.mode: {mode}")

    raw_yaws = views_cfg.get("yaws_deg", (0.0, 90.0, 180.0, 270.0))
    if isinstance(raw_yaws, str):
        yaws_deg = tuple(float(chunk.strip()) for chunk in raw_yaws.split(",") if chunk.strip())
    else:
        yaws_deg = tuple(float(item) for item in raw_yaws)

    return ViewsConfig(
        mode=mode,  # type: ignore[arg-type]
        width=(
            None if views_cfg.get("width") in (None, "") else int(views_cfg["width"])
        ),
        height=(
            None if views_cfg.get("height") in (None, "") else int(views_cfg["height"])
        ),
        fov_deg=(
            None if views_cfg.get("fov_deg") in (None, "") else float(views_cfg["fov_deg"])
        ),
        yaw_deg=float(views_cfg.get("yaw_deg", 0.0)),
        pitch_deg=float(views_cfg.get("pitch_deg", 0.0)),
        roll_deg=float(views_cfg.get("roll_deg", 0.0)),
        yaws_deg=yaws_deg,
        ordering=str(views_cfg.get("ordering", "frame_major")).strip().lower(),  # type: ignore[arg-type]
        rotate_180=bool(views_cfg.get("rotate_180", False)),
        imu_tau=float(views_cfg.get("imu_tau", 0.25)),
        time_offset_ns=int(views_cfg.get("time_offset_ns", 0)),
        use_yaml_timeshift=bool(views_cfg.get("use_yaml_timeshift", False)),
        max_physical_frames=int(views_cfg.get("max_physical_frames", getattr(input_cfg, "max_frames", 0))),
        evaluation_view_index=int(views_cfg.get("evaluation_view_index", 0)),
    )


def _source_cache_payload(sequence: SequenceConfig, profile: ProfileName) -> dict[str, Any]:
    return {
        "profile_limit_physical_frames": requested_physical_frame_limit(sequence, profile),
        "input": asdict(sequence.input),
        "jpeg_quality": sequence.extraction.jpeg_quality,
    }


def _view_cache_payload(sequence: SequenceConfig, profile: ProfileName) -> dict[str, Any]:
    return {
        "profile_limit_physical_frames": requested_physical_frame_limit(sequence, profile),
        "views": asdict(sequence.views),
        "jpeg_quality": sequence.extraction.jpeg_quality,
    }


def _build_view_label(views: ViewsConfig) -> str:
    if views.mode == "equirect":
        return "equirect"
    parts = [views.mode]
    if views.width is not None and views.height is not None:
        parts.append(f"{views.width}x{views.height}")
    if views.fov_deg is not None:
        parts.append(f"fov{int(round(views.fov_deg))}")
    if views.mode == "pinhole_fixed":
        parts.append(f"yaw{int(round(views.yaw_deg))}")
        parts.append(f"pitch{int(round(views.pitch_deg))}")
        parts.append(f"roll{int(round(views.roll_deg))}")
    elif views.mode == "pinhole_level_yaw_imu":
        yaw_label = "-".join(str(int(round(yaw))) for yaw in views.yaws_deg)
        parts.append(f"yaws{yaw_label}")
        parts.append(views.ordering)
    return _slugify("_".join(parts))


def derive_source_cache_root(
    paths: PathsConfig,
    sequence: SequenceConfig,
    profile: ProfileName,
) -> Path:
    source_label = _infer_source_label(sequence.input.source_path)
    prepared_root = paths.outputs_root / "prepared" / source_label
    source_hash = _short_hash(_source_cache_payload(sequence, profile))
    return prepared_root / f"source_{source_hash}"


def derive_view_cache_root(
    paths: PathsConfig,
    sequence: SequenceConfig,
    profile: ProfileName,
    *,
    ordering: ViewOrdering | None = None,
) -> Path:
    source_cache_root = derive_source_cache_root(paths, sequence, profile)
    views = sequence.views if ordering is None else replace(sequence.views, ordering=ordering)
    if views.mode == "equirect":
        return source_cache_root
    view_hash = _short_hash(
        {
            "profile_limit_physical_frames": requested_physical_frame_limit(sequence, profile),
            "views": asdict(views),
            "jpeg_quality": sequence.extraction.jpeg_quality,
        }
    )
    view_label = _build_view_label(views)
    return source_cache_root / "views" / f"{view_label}_{view_hash}"


def load_runner_context(
    paths_config_path: str | Path,
    sequence_config_path: str | Path,
    profile: ProfileName,
) -> RunnerContext:
    if profile not in {"smoke", "full"}:
        raise ValueError(f"Unsupported profile: {profile}")

    paths_cfg = _load_yaml(Path(paths_config_path))
    seq_cfg = _load_yaml(Path(sequence_config_path))

    paths = PathsConfig(
        vggt_root=_as_path(paths_cfg["vggt_root"]),
        hilti_repo_root=_as_path(paths_cfg["hilti_repo_root"]),
        data_root=_as_path(paths_cfg["data_root"]),
        outputs_root=_as_path(paths_cfg["outputs_root"]),
        torch_home=_as_path(paths_cfg["torch_home"]),
        venv_python=_as_path(paths_cfg["venv_python"]),
    )

    extraction_cfg = seq_cfg.get("extraction", {})
    vggt_cfg = seq_cfg.get("vggt", {})
    export_cfg = seq_cfg.get("export", {})
    input_cfg = _parse_input_config(seq_cfg)

    sequence = SequenceConfig(
        run_name=str(seq_cfg["run_name"]),
        input=input_cfg,
        views=_parse_views_config(seq_cfg, input_cfg),
        extraction=ExtractionConfig(
            jpeg_quality=int(extraction_cfg.get("jpeg_quality", 95)),
            smoke_frame_count=int(extraction_cfg.get("smoke_frame_count", 96)),
        ),
        vggt=VggtConfig(
            submap_size=int(vggt_cfg.get("submap_size", 16)),
            overlapping_window_size=int(vggt_cfg.get("overlapping_window_size", 1)),
            max_loops=int(vggt_cfg.get("max_loops", 1)),
            min_disparity=float(vggt_cfg.get("min_disparity", 50.0)),
            conf_threshold=float(vggt_cfg.get("conf_threshold", 25.0)),
            lc_thres=float(vggt_cfg.get("lc_thres", 0.95)),
            disable_flow_keyframes=bool(vggt_cfg.get("disable_flow_keyframes", False)),
            vis_voxel_size=(
                None if vggt_cfg.get("vis_voxel_size") in (None, "") else float(vggt_cfg["vis_voxel_size"])
            ),
        ),
        export=ExportConfig(
            voxel_size=(
                None if export_cfg.get("voxel_size") in (None, "") else float(export_cfg["voxel_size"])
            ),
            nb_neighbors=int(export_cfg.get("nb_neighbors", 20)),
            std_ratio=float(export_cfg.get("std_ratio", 2.0)),
        ),
    )

    source_label = _infer_source_label(sequence.input.source_path)
    prepared_root = paths.outputs_root / "prepared" / source_label
    source_cache_root = derive_source_cache_root(paths, sequence, profile)

    source_frames_dir = source_cache_root / "frames"
    source_frame_manifest_path = source_cache_root / "frame_manifest.csv"
    source_metadata_path = source_cache_root / "source_metadata.yaml"
    source_preview_path = source_cache_root / "frame_preview.jpg"
    stitch_summary_path = source_cache_root / "stitch_summary.yaml"

    if sequence.views.mode == "equirect":
        view_cache_root = source_cache_root
        frames_dir = source_frames_dir
        frame_manifest_path = source_frame_manifest_path
        preview_path = source_preview_path
        view_metadata_path = source_cache_root / "view_metadata.yaml"
    else:
        view_cache_root = derive_view_cache_root(paths, sequence, profile)
        frames_dir = view_cache_root / "frames"
        frame_manifest_path = view_cache_root / "frame_manifest.csv"
        preview_path = view_cache_root / "frame_preview.jpg"
        view_metadata_path = view_cache_root / "view_metadata.yaml"

    run_root = paths.outputs_root / "runs" / sequence.run_name
    smoke_frames_dir = run_root / "smoke_frames"
    profile_root = run_root / profile
    export_name = f"{sequence.run_name}_smoke.ply" if profile == "smoke" else f"{sequence.run_name}.ply"

    layout = LayoutConfig(
        run_root=run_root,
        prepared_root=prepared_root,
        source_cache_root=source_cache_root,
        source_frames_dir=source_frames_dir,
        source_frame_manifest_path=source_frame_manifest_path,
        source_metadata_path=source_metadata_path,
        source_preview_path=source_preview_path,
        stitch_summary_path=stitch_summary_path,
        view_cache_root=view_cache_root,
        frames_dir=frames_dir,
        smoke_frames_dir=smoke_frames_dir,
        frame_manifest_path=frame_manifest_path,
        view_metadata_path=view_metadata_path,
        preview_path=preview_path,
        profile_root=profile_root,
        image_folder=smoke_frames_dir if profile == "smoke" else frames_dir,
        vggt_output_dir=profile_root / "vggt",
        log_path=profile_root / "vggt" / "poses.txt",
        dense_log_dir=profile_root / "vggt" / "poses_logs",
        export_dir=profile_root / "exports",
        export_path=profile_root / "exports" / export_name,
        command_log_path=profile_root / "logs" / "run_vggt.log",
        resolved_config_path=profile_root / "resolved_config.yaml",
        evaluation_inputs_root=profile_root / "evaluation_inputs",
    )

    return RunnerContext(paths=paths, sequence=sequence, layout=layout, profile=profile)


def validate_context(context: RunnerContext) -> None:
    problems: list[str] = []

    if not context.paths.vggt_root.is_dir():
        problems.append(f"VGGT root not found: {context.paths.vggt_root}")
    if not (context.paths.vggt_root / "main.py").is_file():
        problems.append(f"VGGT main.py not found under: {context.paths.vggt_root}")
    if not context.paths.hilti_repo_root.is_dir():
        problems.append(f"Hilti challenge repo not found: {context.paths.hilti_repo_root}")
    if not context.paths.venv_python.is_file():
        problems.append(f"Runner Python executable not found: {context.paths.venv_python}")

    if context.sequence.extraction.smoke_frame_count <= 0:
        problems.append("Smoke frame count must be greater than 0")
    if context.sequence.extraction.jpeg_quality < 1 or context.sequence.extraction.jpeg_quality > 100:
        problems.append("JPEG quality must be within [1, 100]")

    input_cfg = context.sequence.input
    if isinstance(input_cfg, Mp4InputConfig):
        if not input_cfg.source_mp4.is_file():
            problems.append(f"Source MP4 not found: {input_cfg.source_mp4}")
        if input_cfg.sample_fps <= 0:
            problems.append("MP4 input sample_fps must be greater than 0")
    else:
        if not input_cfg.rosbag_db3.is_file():
            problems.append(f"Rosbag DB3 not found: {input_cfg.rosbag_db3}")
        if not input_cfg.calibration_yaml.is_file():
            problems.append(f"Calibration YAML not found: {input_cfg.calibration_yaml}")
        if input_cfg.mask0 is not None and not input_cfg.mask0.is_file():
            problems.append(f"Mask0 not found: {input_cfg.mask0}")
        if input_cfg.mask1 is not None and not input_cfg.mask1.is_file():
            problems.append(f"Mask1 not found: {input_cfg.mask1}")
        if input_cfg.sphere_m <= 0:
            problems.append("Rosbag input sphere_m must be greater than 0")
        if input_cfg.stride <= 0:
            problems.append("Rosbag input stride must be greater than 0")
        if input_cfg.max_frames < 0:
            problems.append("Rosbag input max_frames cannot be negative")
        if input_cfg.sync_tolerance_ns <= 0:
            problems.append("Rosbag input sync_tolerance_ns must be greater than 0")

    views = context.sequence.views
    if views.max_physical_frames < 0:
        problems.append("views.max_physical_frames cannot be negative")
    if views.mode == "equirect":
        if views.ordering != "frame_major":
            problems.append("views.ordering=view_major is only supported for multiview pinhole runs")
    else:
        if views.width is None or views.width <= 0:
            problems.append("Non-equirect view preparation requires views.width > 0")
        if views.height is None or views.height <= 0:
            problems.append("Non-equirect view preparation requires views.height > 0")
        if views.fov_deg is None or not (0.0 < views.fov_deg < 180.0):
            problems.append("Non-equirect view preparation requires 0 < views.fov_deg < 180")
    if views.mode == "pinhole_level_yaw_imu" and len(views.yaws_deg) < 2:
        problems.append("pinhole_level_yaw_imu requires at least two yaw angles")
    if views.ordering == "view_major" and not views.is_multiview:
        problems.append("views.ordering=view_major is only supported for multiview pinhole runs")
    if views.requires_imu and not isinstance(input_cfg, RosbagInputConfig):
        problems.append("IMU-leveled view modes require rosbag input")
    if views.imu_tau <= 0:
        problems.append("views.imu_tau must be greater than 0")
    if views.evaluation_view_index < 0 or views.evaluation_view_index >= views.view_count:
        problems.append("views.evaluation_view_index must reference an existing view")

    if problems:
        raise ValueError("\n".join(problems))


def ensure_layout_dirs(context: RunnerContext) -> None:
    for path in [
        context.paths.outputs_root,
        context.layout.prepared_root,
        context.layout.run_root,
        context.layout.profile_root,
        context.layout.vggt_output_dir,
        context.layout.export_dir,
        context.layout.command_log_path.parent,
        context.layout.evaluation_inputs_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_resolved_config(context: RunnerContext) -> Path:
    context.layout.resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    with context.layout.resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(context.as_dict(), handle, sort_keys=False)
    return context.layout.resolved_config_path
