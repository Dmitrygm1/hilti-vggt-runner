from __future__ import annotations

import csv
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import ViewsConfig


@dataclass(frozen=True)
class FrameManifestRecord:
    frame_index: int
    physical_frame_index: int
    view_index: int
    output_name: str
    source_frame_index: int | None = None
    timestamp_seconds: float | None = None
    cam0_timestamp_ns: int | None = None
    cam1_timestamp_ns: int | None = None
    sync_delta_ns: int | None = None
    yaw_deg: float | None = None
    is_eval_primary: bool = False

    @property
    def source_timestamp_seconds(self) -> float | None:
        if self.cam0_timestamp_ns is not None:
            return float(self.cam0_timestamp_ns) / 1e9
        return self.timestamp_seconds


@dataclass(frozen=True)
class RenderSummary:
    physical_frame_count: int
    emitted_frame_count: int
    output_width: int
    output_height: int
    view_count: int
    ordering: str


def _as_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def read_frame_manifest(path: Path) -> list[FrameManifestRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    records: list[FrameManifestRecord] = []
    for row in rows:
        raw_frame_index = row.get("frame_index") or row.get("extracted_frame_index")
        if raw_frame_index in (None, ""):
            raise ValueError(f"Manifest row is missing frame_index: {row}")
        frame_index = int(raw_frame_index)
        physical_frame_index = int(row.get("physical_frame_index") or frame_index)
        view_index = int(row.get("view_index") or 0)
        output_name = row.get("output_name")
        if output_name in (None, ""):
            raise ValueError(f"Manifest row is missing output_name: {row}")
        is_eval_primary = str(row.get("is_eval_primary", "")).strip().lower() in {"1", "true", "yes"}
        records.append(
            FrameManifestRecord(
                frame_index=frame_index,
                physical_frame_index=physical_frame_index,
                view_index=view_index,
                output_name=output_name,
                source_frame_index=_as_int(row.get("source_frame_index")),
                timestamp_seconds=_as_float(row.get("timestamp_seconds")),
                cam0_timestamp_ns=_as_int(row.get("cam0_timestamp_ns")),
                cam1_timestamp_ns=_as_int(row.get("cam1_timestamp_ns")),
                sync_delta_ns=_as_int(row.get("sync_delta_ns")),
                yaw_deg=_as_float(row.get("yaw_deg")),
                is_eval_primary=is_eval_primary,
            )
        )

    if not records:
        raise RuntimeError(f"No rows found in manifest: {path}")
    return records


def write_frame_manifest(path: Path, records: list[FrameManifestRecord]) -> None:
    fieldnames = [
        "frame_index",
        "physical_frame_index",
        "view_index",
        "source_frame_index",
        "timestamp_seconds",
        "cam0_timestamp_ns",
        "cam1_timestamp_ns",
        "sync_delta_ns",
        "yaw_deg",
        "is_eval_primary",
        "output_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "frame_index": record.frame_index,
                    "physical_frame_index": record.physical_frame_index,
                    "view_index": record.view_index,
                    "source_frame_index": record.source_frame_index,
                    "timestamp_seconds": record.timestamp_seconds,
                    "cam0_timestamp_ns": record.cam0_timestamp_ns,
                    "cam1_timestamp_ns": record.cam1_timestamp_ns,
                    "sync_delta_ns": record.sync_delta_ns,
                    "yaw_deg": record.yaw_deg,
                    "is_eval_primary": int(record.is_eval_primary),
                    "output_name": record.output_name,
                }
            )


def select_records_by_physical_limit(records: list[FrameManifestRecord], physical_limit: int) -> list[FrameManifestRecord]:
    if physical_limit <= 0:
        return list(records)
    selected: list[FrameManifestRecord] = []
    max_physical = None
    for record in records:
        if max_physical is None:
            max_physical = record.physical_frame_index
        if record.physical_frame_index > physical_limit:
            break
        selected.append(record)
    return selected


def build_preview_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    if not frame_paths:
        return

    sample_count = min(4, len(frame_paths))
    sample_indices = np.linspace(0, len(frame_paths) - 1, num=sample_count, dtype=int)
    sampled_images: list[np.ndarray] = []
    for index in sample_indices.tolist():
        image = cv2.imread(str(frame_paths[index]))
        if image is not None:
            sampled_images.append(image)
    if not sampled_images:
        return

    tile_height = min(240, min(image.shape[0] for image in sampled_images))
    resized_tiles: list[np.ndarray] = []
    for image in sampled_images:
        scale = tile_height / image.shape[0]
        tile_width = max(1, int(round(image.shape[1] * scale)))
        resized_tiles.append(cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA))

    max_tile_width = max(tile.shape[1] for tile in resized_tiles)
    padded_tiles: list[np.ndarray] = []
    for tile in resized_tiles:
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


def _read_yaml_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]
    return "\n".join(lines)


def load_cam0_from_yaml(path: Path) -> tuple[np.ndarray, float]:
    data = yaml.safe_load(_read_yaml_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected calibration YAML mapping at {path}")
    cam0 = data["cam0"]
    t_cam_imu = np.array(cam0["T_cam_imu"], dtype=np.float64)
    timeshift = float(cam0.get("timeshift_cam_imu", 0.0))
    return t_cam_imu[:3, :3], timeshift


def parse_imu_packet(data: bytes | memoryview) -> tuple[int, np.ndarray]:
    if isinstance(data, memoryview):
        data = data.tobytes()
    sec, nsec = struct.unpack_from("<iI", data, 4)
    string_len = struct.unpack_from("<I", data, 12)[0]
    pos = 16 + string_len
    pos = (pos + 3) & ~3
    pos += 32  # orientation
    pos += 72  # orientation cov
    pos += 24  # angular velocity
    pos += 72  # angular velocity cov
    linear_acceleration = np.array(struct.unpack_from("<3d", data, pos), dtype=np.float64)
    timestamp_ns = int(sec) * 1_000_000_000 + int(nsec)
    return timestamp_ns, linear_acceleration


def load_imu_series(bag_path: Path, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(str(bag_path))
    try:
        topic = conn.execute(
            "SELECT id FROM topics WHERE name='/imu/data_raw' AND type='sensor_msgs/msg/Imu'"
        ).fetchone()
        if topic is None:
            raise RuntimeError("No /imu/data_raw sensor_msgs/msg/Imu topic found in the bag")

        rows = conn.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic[0],),
        )

        ts_list: list[int] = []
        g_lp_list: list[np.ndarray] = []
        g_lp: np.ndarray | None = None
        last_ts: int | None = None

        for (blob,) in rows:
            ts_ns, accel = parse_imu_packet(blob)
            if g_lp is None or last_ts is None:
                g_lp = accel.astype(np.float64)
            else:
                dt = max((ts_ns - last_ts) * 1e-9, 1e-6)
                alpha = 1.0 - math.exp(-dt / tau_s)
                g_lp = (1.0 - alpha) * g_lp + alpha * accel
            ts_list.append(ts_ns)
            g_lp_list.append(g_lp.copy())
            last_ts = ts_ns
    finally:
        conn.close()

    if not ts_list:
        raise RuntimeError("No IMU messages found in /imu/data_raw")
    return np.asarray(ts_list, dtype=np.int64), np.asarray(g_lp_list, dtype=np.float64)


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return r_yaw @ r_pitch @ r_roll


def yaw_rotation_matrix(yaw_deg: float) -> np.ndarray:
    return rotation_matrix(yaw_deg, 0.0, 0.0)


def build_remap_from_rotation(
    out_w: int,
    out_h: int,
    fov_deg: float,
    rot: np.ndarray,
    in_w: int,
    in_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    fov = math.radians(fov_deg)
    fx = out_w / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    xs, ys = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    z = np.ones_like(x, dtype=np.float32)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_world = dirs @ rot.T

    lon = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = np.arcsin(np.clip(dirs_world[..., 1], -1.0, 1.0))

    map_x = ((lon + math.pi) / (2.0 * math.pi) * in_w).astype(np.float32)
    map_y = ((lat + math.pi / 2.0) / math.pi * in_h).astype(np.float32)
    map_x = np.mod(map_x, in_w).astype(np.float32)
    map_y = np.clip(map_y, 0, in_h - 1).astype(np.float32)
    return map_x, map_y


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector.copy()
    return vector / norm


def level_rotation_from_gravity(g_cam: np.ndarray) -> np.ndarray:
    down = normalize(g_cam)
    if np.linalg.norm(down) < 1e-6:
        return np.eye(3, dtype=np.float32)

    nominal_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = nominal_forward - np.dot(nominal_forward, down) * down
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward = forward - np.dot(forward, down) * down
    forward = normalize(forward)
    right = normalize(np.cross(down, forward))
    rot = np.stack([right, down, forward], axis=1)
    return rot.astype(np.float32)


def _nearest_imu_accel(frame_timestamp_ns: int, imu_timestamps_ns: np.ndarray, imu_gravity_lp: np.ndarray) -> np.ndarray:
    imu_index = int(np.searchsorted(imu_timestamps_ns, frame_timestamp_ns))
    if imu_index >= len(imu_timestamps_ns):
        imu_index = len(imu_timestamps_ns) - 1
    elif imu_index > 0:
        left_dt = abs(int(imu_timestamps_ns[imu_index - 1]) - frame_timestamp_ns)
        right_dt = abs(int(imu_timestamps_ns[imu_index]) - frame_timestamp_ns)
        if left_dt <= right_dt:
            imu_index -= 1
    return imu_gravity_lp[imu_index]


def compute_base_rotations(
    records: list[FrameManifestRecord],
    *,
    bag_path: Path,
    calibration_yaml: Path,
    imu_tau: float,
    time_offset_ns: int,
    use_yaml_timeshift: bool,
) -> dict[int, np.ndarray]:
    r_cam0_imu, yaml_timeshift_s = load_cam0_from_yaml(calibration_yaml)
    imu_timestamps_ns, imu_gravity_lp = load_imu_series(bag_path, tau_s=imu_tau)
    total_offset_ns = int(time_offset_ns)
    if use_yaml_timeshift:
        total_offset_ns += int(round(yaml_timeshift_s * 1e9))

    rotations_by_physical_frame: dict[int, np.ndarray] = {}
    prev_rot = np.eye(3, dtype=np.float32)
    for record in records:
        if record.physical_frame_index in rotations_by_physical_frame:
            continue
        if record.cam0_timestamp_ns is None:
            raise ValueError("IMU-leveled rendering requires cam0_timestamp_ns in the source manifest")
        frame_ts = record.cam0_timestamp_ns + total_offset_ns
        g_imu = _nearest_imu_accel(frame_ts, imu_timestamps_ns, imu_gravity_lp)
        g_cam0 = r_cam0_imu @ g_imu
        rot = level_rotation_from_gravity(g_cam0)
        if not np.isfinite(rot).all():
            rot = prev_rot
        prev_rot = rot
        rotations_by_physical_frame[record.physical_frame_index] = rot
    return rotations_by_physical_frame


def render_views_from_equirect(
    *,
    source_dir: Path,
    source_records: list[FrameManifestRecord],
    output_dir: Path,
    manifest_path: Path,
    preview_path: Path,
    views: ViewsConfig,
    jpeg_quality: int,
    bag_path: Path | None = None,
    calibration_yaml: Path | None = None,
) -> RenderSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(output_dir.iterdir()):
        if child.is_dir():
            raise RuntimeError(f"Expected a flat frame directory, found nested directory: {child}")
        child.unlink()

    source_records = list(source_records)
    first_image = cv2.imread(str(source_dir / source_records[0].output_name), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Failed to read source frame: {source_dir / source_records[0].output_name}")
    in_h, in_w = first_image.shape[:2]
    out_w = in_w if views.width is None else views.width
    out_h = in_h if views.height is None else views.height

    fixed_map_x = fixed_map_y = None
    if views.mode == "pinhole_fixed":
        rot = rotation_matrix(views.yaw_deg, views.pitch_deg, views.roll_deg)
        fixed_map_x, fixed_map_y = build_remap_from_rotation(out_w, out_h, float(views.fov_deg), rot, in_w, in_h)

    rotations_by_physical_frame: dict[int, np.ndarray] = {}
    if views.requires_imu:
        if bag_path is None or calibration_yaml is None:
            raise ValueError("IMU-leveled view generation requires bag_path and calibration_yaml")
        rotations_by_physical_frame = compute_base_rotations(
            source_records,
            bag_path=bag_path,
            calibration_yaml=calibration_yaml,
            imu_tau=views.imu_tau,
            time_offset_ns=views.time_offset_ns,
            use_yaml_timeshift=views.use_yaml_timeshift,
        )

    emitted_records: list[FrameManifestRecord] = []
    current_source_name = None
    current_image = None
    emitted_index = 0

    for source_record in source_records:
        if current_source_name != source_record.output_name:
            current_source_name = source_record.output_name
            current_image = cv2.imread(str(source_dir / source_record.output_name), cv2.IMREAD_COLOR)
            if current_image is None:
                raise RuntimeError(f"Failed to read source frame: {source_dir / source_record.output_name}")

        view_pairs: list[tuple[int, float | None, np.ndarray | tuple[np.ndarray, np.ndarray]]] = []
        if views.mode == "pinhole_fixed":
            assert fixed_map_x is not None and fixed_map_y is not None
            view_pairs.append((0, views.yaw_deg, (fixed_map_x, fixed_map_y)))
        elif views.mode == "pinhole_level_imu":
            base_rot = rotations_by_physical_frame[source_record.physical_frame_index]
            map_x, map_y = build_remap_from_rotation(out_w, out_h, float(views.fov_deg), base_rot, in_w, in_h)
            view_pairs.append((0, 0.0, (map_x, map_y)))
        elif views.mode == "pinhole_level_yaw_imu":
            base_rot = rotations_by_physical_frame[source_record.physical_frame_index]
            for view_index, yaw_deg in enumerate(views.yaws_deg):
                rot = base_rot @ yaw_rotation_matrix(yaw_deg)
                map_x, map_y = build_remap_from_rotation(out_w, out_h, float(views.fov_deg), rot, in_w, in_h)
                view_pairs.append((view_index, yaw_deg, (map_x, map_y)))
        else:
            raise ValueError(f"Unsupported render mode for pinhole rendering: {views.mode}")

        for view_index, yaw_deg, remap in view_pairs:
            map_x, map_y = remap
            pinhole = cv2.remap(
                current_image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            if views.rotate_180:
                pinhole = cv2.rotate(pinhole, cv2.ROTATE_180)

            emitted_index += 1
            output_name = f"frame_{emitted_index:06d}.jpg"
            output_path = output_dir / output_name
            if not cv2.imwrite(str(output_path), pinhole, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise RuntimeError(f"Failed to write rendered frame: {output_path}")

            emitted_records.append(
                FrameManifestRecord(
                    frame_index=emitted_index,
                    physical_frame_index=source_record.physical_frame_index,
                    view_index=view_index,
                    output_name=output_name,
                    source_frame_index=source_record.source_frame_index,
                    timestamp_seconds=source_record.timestamp_seconds,
                    cam0_timestamp_ns=source_record.cam0_timestamp_ns,
                    cam1_timestamp_ns=source_record.cam1_timestamp_ns,
                    sync_delta_ns=source_record.sync_delta_ns,
                    yaw_deg=yaw_deg,
                    is_eval_primary=view_index == views.evaluation_view_index,
                )
            )

    write_frame_manifest(manifest_path, emitted_records)
    build_preview_contact_sheet(sorted(output_dir.iterdir()), preview_path)
    return RenderSummary(
        physical_frame_count=len({record.physical_frame_index for record in emitted_records}),
        emitted_frame_count=len(emitted_records),
        output_width=out_w,
        output_height=out_h,
        view_count=views.view_count,
        ordering="frame_major",
    )


def materialize_view_major_sequence(
    *,
    source_frames_dir: Path,
    source_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
    preview_path: Path,
) -> RenderSummary:
    source_records = read_frame_manifest(source_manifest_path)
    sorted_records = sorted(
        source_records,
        key=lambda record: (record.view_index, record.physical_frame_index, record.frame_index),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(output_dir.iterdir()):
        if child.is_dir():
            raise RuntimeError(f"Expected a flat frame directory, found nested directory: {child}")
        child.unlink()

    emitted_records: list[FrameManifestRecord] = []
    for emitted_index, source_record in enumerate(sorted_records, start=1):
        output_name = f"frame_{emitted_index:06d}.jpg"
        target_path = output_dir / output_name
        target_path.symlink_to(source_frames_dir / source_record.output_name)
        emitted_records.append(
            FrameManifestRecord(
                frame_index=emitted_index,
                physical_frame_index=source_record.physical_frame_index,
                view_index=source_record.view_index,
                output_name=output_name,
                source_frame_index=source_record.source_frame_index,
                timestamp_seconds=source_record.timestamp_seconds,
                cam0_timestamp_ns=source_record.cam0_timestamp_ns,
                cam1_timestamp_ns=source_record.cam1_timestamp_ns,
                sync_delta_ns=source_record.sync_delta_ns,
                yaw_deg=source_record.yaw_deg,
                is_eval_primary=source_record.is_eval_primary,
            )
        )

    write_frame_manifest(manifest_path, emitted_records)
    build_preview_contact_sheet(sorted(output_dir.iterdir()), preview_path)

    first_image = cv2.imread(str(output_dir / emitted_records[0].output_name), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Failed to read reordered frame: {output_dir / emitted_records[0].output_name}")
    out_h, out_w = first_image.shape[:2]
    return RenderSummary(
        physical_frame_count=len({record.physical_frame_index for record in emitted_records}),
        emitted_frame_count=len(emitted_records),
        output_width=out_w,
        output_height=out_h,
        view_count=len({record.view_index for record in emitted_records}),
        ordering="view_major",
    )
