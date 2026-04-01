from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _sanitize(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    return value


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def flatten_metrics(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=full_key))
        else:
            flat[full_key] = _sanitize(value)
    return flat


def write_metrics_csv(path: Path, payload: dict[str, Any]) -> None:
    flat = flatten_metrics(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def write_matched_poses_csv(
    path: Path,
    *,
    timestamps: list[float],
    est_positions: list[list[float]],
    gt_positions: list[list[float]],
    error_xy: list[float],
    error_3d: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "est_x",
                "est_y",
                "est_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "translation_error_xy_m",
                "translation_error_3d_m",
            ],
        )
        writer.writeheader()
        for timestamp, est, gt, xy_error, error_3d in zip(
            timestamps,
            est_positions,
            gt_positions,
            error_xy,
            error_3d,
        ):
            writer.writerow(
                {
                    "timestamp": f"{timestamp:.9f}",
                    "est_x": f"{est[0]:.6f}",
                    "est_y": f"{est[1]:.6f}",
                    "est_z": f"{est[2]:.6f}",
                    "gt_x": f"{gt[0]:.6f}",
                    "gt_y": f"{gt[1]:.6f}",
                    "gt_z": f"{gt[2]:.6f}",
                    "translation_error_xy_m": f"{xy_error:.6f}",
                    "translation_error_3d_m": f"{error_3d:.6f}",
                }
            )


def write_markdown_report(
    path: Path,
    *,
    title: str,
    metadata: dict[str, Any],
    key_metrics: dict[str, Any],
    plots: dict[str, str],
    caveats: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Summary", ""]
    for key, value in metadata.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Key Metrics", ""])
    for key, value in key_metrics.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Plots", ""])
    for label, relative_path in plots.items():
        lines.append(f"- [{label}]({relative_path})")
    if caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
