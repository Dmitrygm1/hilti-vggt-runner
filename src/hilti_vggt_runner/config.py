from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ProfileName = Literal["smoke", "full"]


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
    sample_fps: float
    jpeg_quality: int
    rotate_180: bool
    smoke_frame_count: int


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
    source_mp4: Path
    extraction: ExtractionConfig
    vggt: VggtConfig
    export: ExportConfig


@dataclass(frozen=True)
class LayoutConfig:
    run_root: Path
    frames_dir: Path
    smoke_frames_dir: Path
    frame_manifest_path: Path
    video_metadata_path: Path
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
        source_mp4=_as_path(seq_cfg["source_mp4"]),
        extraction=ExtractionConfig(
            sample_fps=float(extraction_cfg.get("sample_fps", 3.0)),
            jpeg_quality=int(extraction_cfg.get("jpeg_quality", 95)),
            rotate_180=bool(extraction_cfg.get("rotate_180", False)),
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
    export_name = (
        f"{sequence.run_name}_smoke.ply" if profile == "smoke" else f"{sequence.run_name}.ply"
    )

    layout = LayoutConfig(
        run_root=run_root,
        frames_dir=frames_dir,
        smoke_frames_dir=smoke_frames_dir,
        frame_manifest_path=run_root / "frame_manifest.csv",
        video_metadata_path=run_root / "video_metadata.yaml",
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
    if not context.sequence.source_mp4.is_file():
        problems.append(f"Source MP4 not found: {context.sequence.source_mp4}")
    if not context.paths.venv_python.is_file():
        problems.append(f"Runner Python executable not found: {context.paths.venv_python}")
    if context.sequence.extraction.sample_fps <= 0:
        problems.append("Extraction sample_fps must be greater than 0")
    if context.sequence.extraction.smoke_frame_count <= 0:
        problems.append("Smoke frame count must be greater than 0")
    if context.sequence.extraction.jpeg_quality < 1 or context.sequence.extraction.jpeg_quality > 100:
        problems.append("JPEG quality must be within [1, 100]")

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
