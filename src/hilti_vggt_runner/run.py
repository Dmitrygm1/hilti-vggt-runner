from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import RunnerContext, ensure_layout_dirs, write_resolved_config
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
    return command


def _ensure_image_folder_ready(context: RunnerContext) -> None:
    if context.profile == "smoke" and not context.layout.smoke_frames_dir.exists():
        create_smoke_subset(context)

    if context.profile == "full" and context.layout.source_metadata_path.is_file():
        with context.layout.source_metadata_path.open("r", encoding="utf-8") as handle:
            source_metadata = yaml.safe_load(handle) or {}
        if not bool(source_metadata.get("is_complete", False)):
            raise RuntimeError(
                "The current prepared frame set is partial and was likely created for a smoke run.\n"
                "Run prepare_hilti_data.py again with --profile full before launching the full reconstruction."
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
