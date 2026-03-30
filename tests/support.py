from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hilti_vggt_runner.config import RunnerContext, load_runner_context, validate_context


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_context(tmp_path: Path, monkeypatch, *, profile: str = "smoke", source_video_path: Path) -> RunnerContext:
    home_root = tmp_path / "home"
    scratch_root = tmp_path / "scratch"
    monkeypatch.setenv("HOME", str(home_root))
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch_root))

    vggt_root = home_root / "projects" / "VGGT-SLAM"
    vggt_root.mkdir(parents=True)
    (vggt_root / "main.py").write_text("# dummy main\n", encoding="utf-8")

    hilti_root = home_root / "projects" / "hilti-trimble-slam-challenge-2026"
    hilti_root.mkdir(parents=True)

    venv_python = home_root / "jupyter-vggt" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    paths_path = tmp_path / "paths.yaml"
    sequence_path = tmp_path / "sequence.yaml"

    write_yaml(
        paths_path,
        {
            "vggt_root": "${HOME}/projects/VGGT-SLAM",
            "hilti_repo_root": "${HOME}/projects/hilti-trimble-slam-challenge-2026",
            "data_root": "${HOME}/data/hilti-2026",
            "outputs_root": "${SCRATCH_ROOT}/${USER}/hilti-vggt",
            "torch_home": "${SCRATCH_ROOT}/${USER}/torch-cache",
            "venv_python": "${HOME}/jupyter-vggt/bin/python",
        },
    )
    write_yaml(
        sequence_path,
        {
            "run_name": "floor_UG1_2025-06-18_run_1",
            "source_mp4": str(source_video_path),
            "extraction": {
                "sample_fps": 1.0,
                "jpeg_quality": 95,
                "rotate_180": True,
                "smoke_frame_count": 1,
            },
            "vggt": {
                "submap_size": 16,
                "overlapping_window_size": 1,
                "max_loops": 1,
                "min_disparity": 50.0,
                "conf_threshold": 25.0,
                "lc_thres": 0.95,
            },
            "export": {
                "voxel_size": None,
                "nb_neighbors": 0,
                "std_ratio": 0.0,
            },
        },
    )

    context = load_runner_context(paths_path, sequence_path, profile=profile)
    validate_context(context)
    return context
