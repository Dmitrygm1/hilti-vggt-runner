from __future__ import annotations

import csv
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np
import yaml

JPEG_SOI = bytes([0xFF, 0xD8, 0xFF])


@dataclass(frozen=True)
class RosbagStitchConfig:
    bag_path: Path
    out_dir: Path
    frame_manifest_path: Path
    stitch_summary_path: Path
    preview_path: Path
    calibration_yaml: Path
    mask0: Path | None
    mask1: Path | None
    jpeg_quality: int
    sphere_m: float
    stride: int
    max_frames: int
    rotate_180: bool
    sync_tolerance_ns: int
    topic0: str
    topic1: str


@dataclass
class PairingStats:
    synced_pairs: int = 0
    advanced_cam0: int = 0
    advanced_cam1: int = 0


@dataclass(frozen=True)
class RosbagExtractionSummary:
    extracted_frame_count: int
    output_width: int
    output_height: int
    paired_messages: int
    sampled_pairs: int
    advanced_cam0: int
    advanced_cam1: int


@dataclass(frozen=True)
class _PrecomputedStitcher:
    maps0_cv: tuple[np.ndarray, np.ndarray]
    maps1_cv: tuple[np.ndarray, np.ndarray]
    w0: np.ndarray
    w1: np.ndarray
    mask_none: np.ndarray
    output_width: int
    output_height: int


@dataclass(frozen=True)
class SyncedPair:
    cam0_timestamp_ns: int
    cam1_timestamp_ns: int
    raw0: bytes
    raw1: bytes


def _read_yaml_text(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]
    return "\n".join(lines)


def load_camera_yaml(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = yaml.safe_load(_read_yaml_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"Calibration YAML did not parse to a mapping: {path}")
    return data["cam0"], data["cam1"]


def parse_intrinsics(cam: dict[str, Any]) -> Dict[str, Any]:
    model = str(cam.get("camera_model", "")).strip().lower()
    resolution = cam.get("resolution")
    if resolution is None:
        raise ValueError("Camera calibration is missing resolution")
    w, h = int(resolution[0]), int(resolution[1])
    intr = cam.get("intrinsics")
    if intr is None:
        raise ValueError("Camera calibration is missing intrinsics")

    if model.startswith("eucm"):
        if len(intr) < 6:
            raise ValueError("EUCM intrinsics must be [alpha, beta, fx, fy, cx, cy]")
        alpha, beta, fx, fy, cx, cy = (float(x) for x in intr[:6])
        fov_deg = float(cam.get("fov_deg", 190.0))
        return {
            "model": "eucm",
            "alpha": alpha,
            "beta": beta,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "k": np.zeros(0, dtype=np.float64),
            "max_theta": math.radians(fov_deg / 2.0),
        }

    if len(intr) < 4:
        raise ValueError("KB intrinsics must be [fx, fy, cx, cy]")
    distortion = cam.get("distortion_coeffs") or []
    fov_deg = float(cam.get("fov_deg", 195.0))
    return {
        "model": "kb",
        "fx": float(intr[0]),
        "fy": float(intr[1]),
        "cx": float(intr[2]),
        "cy": float(intr[3]),
        "w": w,
        "h": h,
        "k": np.array(distortion, dtype=np.float64),
        "max_theta": math.radians(fov_deg / 2.0),
    }


def resolve_extrinsics(cam0: dict[str, Any], cam1: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    stitch_transform = cam1.get("T_stitch")
    if stitch_transform is not None:
        matrix = np.array(stitch_transform, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix[:3, :3], matrix[:3, 3]

    def _parse_t_cam_imu(cam: dict[str, Any]) -> np.ndarray | None:
        matrix = cam.get("T_cam_imu")
        if matrix is None:
            return None
        parsed = np.array(matrix, dtype=np.float64)
        return parsed if parsed.shape == (4, 4) else None

    transform0 = _parse_t_cam_imu(cam0)
    transform1 = _parse_t_cam_imu(cam1)
    if transform0 is not None and transform1 is not None:
        relative = transform1 @ np.linalg.inv(transform0)
        return relative[:3, :3], relative[:3, 3]

    return np.eye(3), np.zeros(3)


def _rho_from_theta(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    rho = theta.copy()
    if k.size == 0:
        return rho
    theta_sq = theta * theta
    theta_power = theta * theta_sq
    for coefficient in k.flat:
        rho = rho + coefficient * theta_power
        theta_power = theta_power * theta_sq
    return rho


def dirs_to_pixels_kb(dirs: np.ndarray, intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    rho_xy = np.hypot(x, y)
    theta = np.arctan2(rho_xy, z)
    focal = 0.5 * (intrinsics["fx"] + intrinsics["fy"])
    rho_px = focal * _rho_from_theta(theta, intrinsics["k"])
    nonzero = rho_xy > 1e-12
    unit_x = np.zeros_like(rho_xy)
    unit_y = np.zeros_like(rho_xy)
    unit_x[nonzero] = x[nonzero] / rho_xy[nonzero]
    unit_y[nonzero] = y[nonzero] / rho_xy[nonzero]
    return intrinsics["cx"] + unit_x * rho_px, intrinsics["cy"] + unit_y * rho_px, theta


def dirs_to_pixels_eucm(dirs: np.ndarray, intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    norm = np.sqrt(x**2 + y**2 + z**2)
    xi = float(intrinsics["alpha"])
    denom = np.where(np.abs(z + xi * norm) < 1e-12, 1e-12, z + xi * norm)
    theta = np.arctan2(np.hypot(x, y), z)
    return intrinsics["fx"] * x / denom + intrinsics["cx"], intrinsics["fy"] * y / denom + intrinsics["cy"], theta


def build_equirect_dirs(out_w: int, out_h: int) -> np.ndarray:
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    lon = xs[None, :] * 2.0 * np.pi - np.pi
    lat = ys[:, None] * np.pi - (np.pi / 2.0)
    lon = np.broadcast_to(lon, (out_h, out_w))
    lat = np.broadcast_to(lat, (out_h, out_w))
    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def build_remap(
    intrinsics: dict[str, Any],
    dirs: np.ndarray,
    out_h: int,
    out_w: int,
    sphere_r: float,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    points = dirs * sphere_r
    projector = dirs_to_pixels_eucm if intrinsics["model"] == "eucm" else dirs_to_pixels_kb
    u, v, theta = projector(points.astype(np.float64), intrinsics)
    width, height = int(intrinsics["w"]), int(intrinsics["h"])
    map_x = u.reshape(out_h, out_w).astype(np.float32)
    map_y = v.reshape(out_h, out_w).astype(np.float32)
    valid = (
        (theta <= intrinsics["max_theta"] + 1e-9).reshape(out_h, out_w)
        & (map_x >= 0)
        & (map_x < width - 1)
        & (map_y >= 0)
        & (map_y < height - 1)
    )
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2), map_x, map_y, valid


def compute_blend_weights(
    valid0: np.ndarray,
    dist0: np.ndarray,
    valid1: np.ndarray,
    dist1: np.ndarray,
    mask0: np.ndarray | None,
    mask1: np.ndarray | None,
    maps0: tuple[np.ndarray, np.ndarray],
    maps1: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map0_1, map0_2 = maps0
    map1_1, map1_2 = maps1

    def _apply_mask(valid: np.ndarray, map1: np.ndarray, map2: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return valid
        remapped = cv2.remap(
            mask,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return valid & (remapped != 255)

    valid0_masked = _apply_mask(valid0, map0_1, map0_2, mask0)
    valid1_masked = _apply_mask(valid1, map1_1, map1_2, mask1)

    eps = 1e-6
    dist0_masked = np.where(valid0_masked, np.maximum(dist0, eps), 0.0)
    dist1_masked = np.where(valid1_masked, np.maximum(dist1, eps), 0.0)
    inv0 = np.divide(1.0, dist0_masked, where=dist0_masked > 0, out=np.zeros_like(dist0_masked))
    inv1 = np.divide(1.0, dist1_masked, where=dist1_masked > 0, out=np.zeros_like(dist1_masked))
    weight_sum = inv0 + inv1 + 1e-12
    weight0 = (inv0 / weight_sum).astype(np.float32)
    weight1 = (inv1 / weight_sum).astype(np.float32)
    mask_none = (inv0 + inv1) == 0
    return weight0, weight1, mask_none


def maybe_rotate_image(image: np.ndarray, rotate_180: bool) -> np.ndarray:
    if not rotate_180:
        return image
    return cv2.rotate(image, cv2.ROTATE_180)


def stitch(
    img0: np.ndarray,
    img1: np.ndarray,
    maps0: tuple[np.ndarray, np.ndarray],
    maps1: tuple[np.ndarray, np.ndarray],
    w0: np.ndarray,
    w1: np.ndarray,
    mask_none: np.ndarray,
) -> np.ndarray:
    map0_1, map0_2 = maps0
    map1_1, map1_2 = maps1

    def _remap(source: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
        return cv2.remap(
            source,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        remapped0 = executor.submit(_remap, img0, map0_1, map0_2)
        remapped1 = executor.submit(_remap, img1, map1_1, map1_2)
        image0, image1 = remapped0.result(), remapped1.result()

    output = image0.astype(np.float32) * w0[..., None] + image1.astype(np.float32) * w1[..., None]
    output[mask_none] = 0.0
    return np.clip(output, 0, 255).astype(np.uint8)


def get_topic_id(conn: sqlite3.Connection, name: str) -> int | None:
    row = conn.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
    return int(row[0]) if row else None


def find_jpeg(data: bytes) -> bytes | None:
    start = data.find(JPEG_SOI)
    return data[start:] if start != -1 else None


def iter_synchronized_pairs(
    conn: sqlite3.Connection,
    topic0_id: int,
    topic1_id: int,
    sync_tolerance_ns: int,
    stats: PairingStats | None = None,
) -> Iterator[SyncedPair]:
    cursor0 = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic0_id,),
    )
    cursor1 = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic1_id,),
    )

    buf0 = cursor0.fetchone()
    buf1 = cursor1.fetchone()
    pairing_stats = stats if stats is not None else PairingStats()

    while buf0 and buf1:
        timestamp0, raw0 = buf0
        timestamp1, raw1 = buf1
        delta = int(timestamp0) - int(timestamp1)

        if abs(delta) <= sync_tolerance_ns:
            pairing_stats.synced_pairs += 1
            yield SyncedPair(
                cam0_timestamp_ns=int(timestamp0),
                cam1_timestamp_ns=int(timestamp1),
                raw0=bytes(raw0) if isinstance(raw0, memoryview) else raw0,
                raw1=bytes(raw1) if isinstance(raw1, memoryview) else raw1,
            )
            buf0 = cursor0.fetchone()
            buf1 = cursor1.fetchone()
        elif delta < 0:
            pairing_stats.advanced_cam0 += 1
            buf0 = cursor0.fetchone()
        else:
            pairing_stats.advanced_cam1 += 1
            buf1 = cursor1.fetchone()


def _load_mask(path: Path | None, intrinsics: dict[str, Any]) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to load mask image: {path}")
    return cv2.resize(mask, (int(intrinsics["w"]), int(intrinsics["h"])), interpolation=cv2.INTER_NEAREST)


def build_stitcher(config: RosbagStitchConfig) -> _PrecomputedStitcher:
    cam0_cfg, cam1_cfg = load_camera_yaml(config.calibration_yaml)
    intrinsics0 = parse_intrinsics(cam0_cfg)
    intrinsics1 = parse_intrinsics(cam1_cfg)
    rotation_01, translation_01 = resolve_extrinsics(cam0_cfg, cam1_cfg)

    output_width = max(1024, 2 * min(int(intrinsics0["w"]), int(intrinsics1["w"])))
    output_height = output_width // 2
    dirs = build_equirect_dirs(output_width, output_height).astype(np.float64)
    cam1_points = (dirs * config.sphere_m - translation_01) @ rotation_01.T

    maps0_cv, _, _, valid0 = build_remap(intrinsics0, dirs, output_height, output_width, config.sphere_m)
    maps1_cv, _, _, valid1 = build_remap(intrinsics1, cam1_points, output_height, output_width, config.sphere_m)

    dist0 = np.linalg.norm(dirs * config.sphere_m, axis=1).reshape(output_height, output_width)
    dist1 = np.linalg.norm(cam1_points, axis=1).reshape(output_height, output_width)

    mask0 = _load_mask(config.mask0, intrinsics0)
    mask1 = _load_mask(config.mask1, intrinsics1)
    w0, w1, mask_none = compute_blend_weights(valid0, dist0, valid1, dist1, mask0, mask1, maps0_cv, maps1_cv)

    return _PrecomputedStitcher(
        maps0_cv=maps0_cv,
        maps1_cv=maps1_cv,
        w0=w0,
        w1=w1,
        mask_none=mask_none,
        output_width=output_width,
        output_height=output_height,
    )


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        for child in sorted(path.iterdir()):
            if child.is_dir():
                raise RuntimeError(f"Expected a flat frame directory, found nested directory: {child}")
            child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)


def _build_preview(frame_paths: list[Path], preview_path: Path) -> None:
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

    row_width = max(tile.shape[1] for tile in resized_tiles)
    padded_tiles: list[np.ndarray] = []
    for tile in resized_tiles:
        if tile.shape[1] < row_width:
            pad = np.zeros((tile.shape[0], row_width - tile.shape[1], 3), dtype=tile.dtype)
            tile = np.hstack([tile, pad])
        padded_tiles.append(tile)

    if len(padded_tiles) == 1:
        contact_sheet = padded_tiles[0]
    else:
        while len(padded_tiles) < 4:
            padded_tiles.append(np.zeros_like(padded_tiles[0]))
        top = np.hstack([padded_tiles[0], padded_tiles[1]])
        bottom = np.hstack([padded_tiles[2], padded_tiles[3]])
        contact_sheet = np.vstack([top, bottom])

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def extract_rosbag_frames(config: RosbagStitchConfig) -> RosbagExtractionSummary:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.frame_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.stitch_summary_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_clean_directory(config.out_dir)

    stitcher = build_stitcher(config)
    pairing_stats = PairingStats()
    manifest_rows: list[dict[str, int | str]] = []
    frame_paths: list[Path] = []
    sampled_pairs = 0
    extracted_count = 0

    conn = sqlite3.connect(str(config.bag_path))
    try:
        topic0_id = get_topic_id(conn, config.topic0)
        topic1_id = get_topic_id(conn, config.topic1)
        if topic0_id is None or topic1_id is None:
            raise RuntimeError(
                "Required camera topics were not found in the bag: "
                f"{config.topic0!r}, {config.topic1!r}"
            )

        for pair_index, pair in enumerate(
            iter_synchronized_pairs(conn, topic0_id, topic1_id, config.sync_tolerance_ns, stats=pairing_stats),
            start=1,
        ):
            if (pair_index - 1) % config.stride != 0:
                continue

            jpeg0 = find_jpeg(pair.raw0)
            jpeg1 = find_jpeg(pair.raw1)
            if jpeg0 is None or jpeg1 is None:
                continue

            img0 = cv2.imdecode(np.frombuffer(jpeg0, dtype=np.uint8), cv2.IMREAD_COLOR)
            img1 = cv2.imdecode(np.frombuffer(jpeg1, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img0 is None or img1 is None:
                continue

            stitched = stitch(
                img0,
                img1,
                stitcher.maps0_cv,
                stitcher.maps1_cv,
                stitcher.w0,
                stitcher.w1,
                stitcher.mask_none,
            )
            stitched = maybe_rotate_image(stitched, config.rotate_180)

            extracted_count += 1
            sampled_pairs += 1
            output_name = f"frame_{extracted_count:06d}.jpg"
            output_path = config.out_dir / output_name
            success = cv2.imwrite(
                str(output_path),
                stitched,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
            )
            if not success:
                raise RuntimeError(f"Failed to write stitched frame: {output_path}")

            frame_paths.append(output_path)
            manifest_rows.append(
                {
                    "frame_index": extracted_count,
                    "cam0_timestamp_ns": pair.cam0_timestamp_ns,
                    "cam1_timestamp_ns": pair.cam1_timestamp_ns,
                    "sync_delta_ns": pair.cam0_timestamp_ns - pair.cam1_timestamp_ns,
                    "output_name": output_name,
                }
            )

            if config.max_frames > 0 and extracted_count >= config.max_frames:
                break
    finally:
        conn.close()

    with config.frame_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "cam0_timestamp_ns", "cam1_timestamp_ns", "sync_delta_ns", "output_name"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    _build_preview(frame_paths, config.preview_path)

    with config.stitch_summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "bag_path": str(config.bag_path),
                "topic0": config.topic0,
                "topic1": config.topic1,
                "output_width": stitcher.output_width,
                "output_height": stitcher.output_height,
                "stride": config.stride,
                "max_frames": config.max_frames,
                "rotate_180": config.rotate_180,
                "sphere_m": config.sphere_m,
                "sync_tolerance_ns": config.sync_tolerance_ns,
                "paired_messages": pairing_stats.synced_pairs,
                "sampled_pairs": sampled_pairs,
                "extracted_frames": extracted_count,
                "advanced_cam0": pairing_stats.advanced_cam0,
                "advanced_cam1": pairing_stats.advanced_cam1,
                "preview_path": str(config.preview_path),
            },
            handle,
            sort_keys=False,
        )

    return RosbagExtractionSummary(
        extracted_frame_count=extracted_count,
        output_width=stitcher.output_width,
        output_height=stitcher.output_height,
        paired_messages=pairing_stats.synced_pairs,
        sampled_pairs=sampled_pairs,
        advanced_cam0=pairing_stats.advanced_cam0,
        advanced_cam1=pairing_stats.advanced_cam1,
    )
