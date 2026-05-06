from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import RunnerContext, ensure_layout_dirs, requested_physical_frame_limit, write_resolved_config
from .prepare import create_smoke_subset, list_image_files


@dataclass(frozen=True)
class RunSummary:
    command: list[str]
    log_path: Path
    dense_log_dir: Path
    poses_path: Path


def suggest_gpu_shell_command() -> str:
    return "srun --pty -A 3dv --gpus=5060ti:1 -t 120 bash --login"


def probe_cuda(python_executable: Path) -> dict[str, object]:
    probe_code = (
        "import json, torch; "
        "print(json.dumps({'cuda_available': bool(torch.cuda.is_available()), 'device_count': torch.cuda.device_count()}))"
    )
    result = subprocess.run(
        [str(python_executable), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def build_vggt_command(context: RunnerContext) -> list[str]:
    command = [
        str(context.paths.venv_python),
        "main.py",
        "--image_folder",
        str(context.layout.image_folder),
        "--headless",
        "--log_results",
        "--log_path",
        str(context.layout.log_path),
        "--submap_size",
        str(context.sequence.vggt.submap_size),
        "--overlapping_window_size",
        str(context.sequence.vggt.overlapping_window_size),
        "--max_loops",
        str(context.sequence.vggt.max_loops),
        "--min_disparity",
        str(context.sequence.vggt.min_disparity),
        "--conf_threshold",
        str(context.sequence.vggt.conf_threshold),
        "--lc_thres",
        str(context.sequence.vggt.lc_thres),
    ]
    if context.sequence.vggt.vis_voxel_size is not None:
        command.extend(["--vis_voxel_size", str(context.sequence.vggt.vis_voxel_size)])
    if context.sequence.vggt.disable_flow_keyframes:
        command.append("--disable_flow_keyframes")
    return command


def _as_positive_int(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def metadata_matches_configured_full_limit(context: RunnerContext, metadata: dict[str, object]) -> bool:
    requested_limit = requested_physical_frame_limit(context.sequence, context.profile)
    if requested_limit <= 0:
        return False

    metadata_limit = _as_positive_int(metadata.get("requested_physical_frame_limit"))
    if metadata_limit != requested_limit:
        return False

    physical_frames = _as_positive_int(metadata.get("physical_frames")) or _as_positive_int(metadata.get("extracted_frames"))
    if physical_frames < requested_limit:
        return False

    extracted_frames = _as_positive_int(metadata.get("extracted_frames"))
    view_count = _as_positive_int(metadata.get("view_count")) or context.sequence.views.view_count
    if "physical_frames" in metadata and extracted_frames < requested_limit * view_count:
        return False

    return True


def _ensure_image_folder_ready(context: RunnerContext) -> None:
    if context.profile == "smoke" and not context.layout.smoke_frames_dir.exists():
        create_smoke_subset(context)

    metadata_path = context.layout.view_metadata_path if context.layout.view_metadata_path.is_file() else context.layout.source_metadata_path
    if context.profile == "full" and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            preparation_metadata = yaml.safe_load(handle) or {}
        if not bool(preparation_metadata.get("is_complete", False)) and not metadata_matches_configured_full_limit(
            context,
            preparation_metadata,
        ):
            raise RuntimeError(
                "The current prepared frame set is partial and was likely created for a smoke run.\n"
                "Run prepare_hilti_data.py again with --profile full before launching the full reconstruction.\n"
                "If this is an intentionally capped full run, make sure views.max_physical_frames matches the prepared metadata."
            )

    image_files = list_image_files(context.layout.image_folder)
    if not image_files:
        raise RuntimeError(f"No input images found in {context.layout.image_folder}. Run prepare_hilti_data.py first.")


def run_vggt(context: RunnerContext, allow_cpu: bool = False) -> RunSummary:
    ensure_layout_dirs(context)
    write_resolved_config(context)
    _ensure_image_folder_ready(context)

    cuda_info = probe_cuda(context.paths.venv_python)
    if not cuda_info["cuda_available"] and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available in the current shell.\n"
            f"Start a GPU shell first, for example:\n{suggest_gpu_shell_command()}"
        )

    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(context.paths.torch_home))
    env["PYTHONUNBUFFERED"] = "1"

    context.layout.command_log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_vggt_command(context)

    with context.layout.command_log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("Command:\n")
        log_handle.write(" ".join(command))
        log_handle.write("\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(context.paths.vggt_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"VGGT-SLAM failed with exit code {return_code}. See {context.layout.command_log_path}")
    if not context.layout.log_path.is_file():
        raise RuntimeError(f"Expected pose log not found: {context.layout.log_path}")
    if not context.layout.dense_log_dir.is_dir():
        raise RuntimeError(f"Expected dense log directory not found: {context.layout.dense_log_dir}")
    if not any(context.layout.dense_log_dir.glob("*.npz")):
        raise RuntimeError(f"No framewise pointcloud logs were written to {context.layout.dense_log_dir}")

    return RunSummary(
        command=command,
        log_path=context.layout.command_log_path,
        dense_log_dir=context.layout.dense_log_dir,
        poses_path=context.layout.log_path,
    )
