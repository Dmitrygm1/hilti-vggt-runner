from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ProfileName = Literal["smoke", "full"]
InputType = Literal["mp4", "rosbag"]


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
    return value


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
class VggtConfig:
    submap_size: int
    overlapping_window_size: int
    max_loops: int
    min_disparity: float
    conf_threshold: float
    lc_thres: float
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
    extraction: ExtractionConfig
    vggt: VggtConfig
    export: ExportConfig


@dataclass(frozen=True)
class LayoutConfig:
    run_root: Path
    frames_dir: Path
    smoke_frames_dir: Path
    frame_manifest_path: Path
    source_metadata_path: Path
    stitch_summary_path: Path
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

    sequence = SequenceConfig(
        run_name=str(seq_cfg["run_name"]),
        input=_parse_input_config(seq_cfg),
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

    run_root = paths.outputs_root / "runs" / sequence.run_name
    frames_dir = run_root / "frames"
    smoke_frames_dir = run_root / "smoke_frames"
    profile_root = run_root / profile
    export_name = f"{sequence.run_name}_smoke.ply" if profile == "smoke" else f"{sequence.run_name}.ply"

    layout = LayoutConfig(
        run_root=run_root,
        frames_dir=frames_dir,
        smoke_frames_dir=smoke_frames_dir,
        frame_manifest_path=run_root / "frame_manifest.csv",
        source_metadata_path=run_root / "source_metadata.yaml",
        stitch_summary_path=run_root / "stitch_summary.yaml",
        preview_path=run_root / "frame_preview.jpg",
        profile_root=profile_root,
        image_folder=smoke_frames_dir if profile == "smoke" else frames_dir,
        vggt_output_dir=profile_root / "vggt",
        log_path=profile_root / "vggt" / "poses.txt",
        dense_log_dir=profile_root / "vggt" / "poses_logs",
        export_dir=profile_root / "exports",
        export_path=profile_root / "exports" / export_name,
        command_log_path=profile_root / "logs" / "run_vggt.log",
        resolved_config_path=profile_root / "resolved_config.yaml",
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

    if problems:
        raise ValueError("\n".join(problems))


def ensure_layout_dirs(context: RunnerContext) -> None:
    for path in [
        context.paths.outputs_root,
        context.layout.run_root,
        context.layout.profile_root,
        context.layout.vggt_output_dir,
        context.layout.export_dir,
        context.layout.command_log_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_resolved_config(context: RunnerContext) -> Path:
    context.layout.resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    with context.layout.resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(context.as_dict(), handle, sort_keys=False)
    return context.layout.resolved_config_path
