from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..views import FrameManifestRecord, read_frame_manifest, write_frame_manifest


@dataclass(frozen=True)
class DerivedEvaluationInputs:
    output_root: Path
    resolved_config_path: Path
    poses_path: Path
    manifest_path: Path
    selected_view_index: int
    selected_frame_count: int
    physical_frame_count: int


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at {path}")
    return loaded


def _detect_default_view_index(records: list[FrameManifestRecord]) -> int:
    primary = [record.view_index for record in records if record.is_eval_primary]
    if primary:
        return primary[0]
    return min(record.view_index for record in records)


def _is_multiview(records: list[FrameManifestRecord]) -> bool:
    return len({record.view_index for record in records}) > 1


def derive_multiview_evaluation_inputs(
    resolved_config_path: Path,
    *,
    view_index: int | None = None,
) -> DerivedEvaluationInputs:
    resolved_config_path = resolved_config_path.expanduser().resolve()
    resolved_payload = _load_yaml(resolved_config_path)
    layout = resolved_payload.get("layout") or {}
    if not isinstance(layout, dict):
        raise ValueError("Resolved config is missing the layout mapping")

    manifest_path = Path(layout["frame_manifest_path"]).expanduser().resolve()
    poses_path = Path(layout["log_path"]).expanduser().resolve()
    profile_root = Path(layout["profile_root"]).expanduser().resolve()

    records = read_frame_manifest(manifest_path)
    selected_view_index = _detect_default_view_index(records) if view_index is None else int(view_index)
    selected_records = [record for record in records if record.view_index == selected_view_index]
    if not selected_records:
        raise RuntimeError(
            f"No manifest rows found for view_index={selected_view_index} in {manifest_path}"
        )

    output_root = profile_root / "evaluation_inputs" / f"view_{selected_view_index:02d}"
    output_root.mkdir(parents=True, exist_ok=True)

    derived_manifest_path = output_root / "frame_manifest.csv"
    write_frame_manifest(derived_manifest_path, selected_records)

    selected_frame_ids = {record.frame_index for record in selected_records}
    derived_poses_path = output_root / "poses.txt"
    with poses_path.open("r", encoding="utf-8") as src, derived_poses_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            frame_id = int(float(parts[0]))
            if frame_id in selected_frame_ids:
                dst.write(stripped)
                dst.write("\n")

    if not derived_poses_path.is_file() or derived_poses_path.stat().st_size == 0:
        raise RuntimeError(
            f"No pose rows matched the selected multiview subset for view_index={selected_view_index}"
        )

    derived_resolved = resolved_payload
    derived_layout = dict(layout)
    derived_layout["frame_manifest_path"] = str(derived_manifest_path)
    derived_layout["log_path"] = str(derived_poses_path)
    derived_resolved["layout"] = derived_layout

    resolved_out = output_root / "resolved_config.yaml"
    with resolved_out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(derived_resolved, handle, sort_keys=False)

    with (output_root / "selection.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "source_resolved_config": str(resolved_config_path),
                "selected_view_index": selected_view_index,
                "selected_frame_count": len(selected_records),
                "physical_frame_count": len({record.physical_frame_index for record in selected_records}),
                "is_multiview_source": _is_multiview(records),
            },
            handle,
            sort_keys=False,
        )

    return DerivedEvaluationInputs(
        output_root=output_root,
        resolved_config_path=resolved_out,
        poses_path=derived_poses_path,
        manifest_path=derived_manifest_path,
        selected_view_index=selected_view_index,
        selected_frame_count=len(selected_records),
        physical_frame_count=len({record.physical_frame_index for record in selected_records}),
    )
