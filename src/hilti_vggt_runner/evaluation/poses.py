from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class PoseSequence:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    frame_ids: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> "PoseSequence":
        return PoseSequence(
            timestamps=self.timestamps[mask],
            positions=self.positions[mask],
            quaternions_xyzw=self.quaternions_xyzw[mask],
            frame_ids=None if self.frame_ids is None else self.frame_ids[mask],
        )

    @property
    def count(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass(frozen=True)
class EstimatedPoseLoadSummary:
    raw_rows: int
    deduped_rows: int
    duplicate_rows: int
    identical_duplicate_rows: int
    conflicting_duplicate_rows: int


@dataclass(frozen=True)
class InitPose:
    run_name: str
    floorplan_name: str
    timestamp: float
    position: np.ndarray
    quaternion_xyzw: np.ndarray


def load_tum_pose_sequence(path: Path) -> PoseSequence:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            timestamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            quaternions.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not timestamps:
        raise RuntimeError(f"No poses found in {path}")
    return PoseSequence(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        quaternions_xyzw=np.asarray(quaternions, dtype=np.float64),
    )


def _load_manifest_timestamps(
    manifest_path: Path,
    *,
    input_type: str,
    absolute_start_time_seconds: float | None,
) -> dict[int, float]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in frame manifest: {manifest_path}")

    timestamps_by_frame: dict[int, float] = {}
    for row in rows:
        raw_frame_id = row.get("frame_index") or row.get("extracted_frame_index")
        if raw_frame_id in (None, ""):
            raise ValueError(f"Missing frame index column in {manifest_path}")
        frame_id = int(raw_frame_id)

        if input_type == "rosbag":
            raw_ts_ns = row.get("cam0_timestamp_ns")
            if raw_ts_ns in (None, ""):
                raise ValueError(f"Rosbag manifest missing cam0_timestamp_ns in {manifest_path}")
            timestamps_by_frame[frame_id] = float(raw_ts_ns) / 1e9
        elif input_type == "mp4":
            raw_ts_seconds = row.get("timestamp_seconds")
            if raw_ts_seconds in (None, ""):
                raise ValueError(f"MP4 manifest missing timestamp_seconds in {manifest_path}")
            base = 0.0 if absolute_start_time_seconds is None else absolute_start_time_seconds
            timestamps_by_frame[frame_id] = base + float(raw_ts_seconds)
        else:
            raise ValueError(f"Unsupported input_type for manifest timestamps: {input_type}")

    return timestamps_by_frame


def load_estimated_pose_sequence(
    poses_path: Path,
    manifest_path: Path,
    *,
    input_type: str,
    absolute_start_time_seconds: float | None = None,
) -> tuple[PoseSequence, EstimatedPoseLoadSummary]:
    timestamps_by_frame = _load_manifest_timestamps(
        manifest_path,
        input_type=input_type,
        absolute_start_time_seconds=absolute_start_time_seconds,
    )

    rows_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    duplicate_rows = 0
    identical_duplicates = 0
    conflicting_duplicates = 0
    raw_rows = 0

    with poses_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [float(token) for token in line.split()]
            if len(parts) < 8:
                continue
            raw_rows += 1
            frame_id = int(parts[0])
            position = np.asarray(parts[1:4], dtype=np.float64)
            quaternion = np.asarray(parts[4:8], dtype=np.float64)
            previous = rows_by_frame.get(frame_id)
            if previous is not None:
                duplicate_rows += 1
                if np.allclose(previous[0], position) and np.allclose(previous[1], quaternion):
                    identical_duplicates += 1
                else:
                    conflicting_duplicates += 1
            rows_by_frame[frame_id] = (position, quaternion)

    if not rows_by_frame:
        raise RuntimeError(f"No estimated poses found in {poses_path}")

    missing_frame_ids = sorted(frame_id for frame_id in rows_by_frame if frame_id not in timestamps_by_frame)
    if missing_frame_ids:
        preview = ", ".join(str(frame_id) for frame_id in missing_frame_ids[:10])
        raise RuntimeError(
            f"Estimated poses reference frame IDs that are missing from the manifest {manifest_path}: {preview}"
        )

    frame_ids = np.asarray(sorted(rows_by_frame), dtype=np.int32)
    positions = np.vstack([rows_by_frame[frame_id][0] for frame_id in frame_ids])
    quaternions = np.vstack([rows_by_frame[frame_id][1] for frame_id in frame_ids])
    timestamps = np.asarray([timestamps_by_frame[int(frame_id)] for frame_id in frame_ids], dtype=np.float64)

    summary = EstimatedPoseLoadSummary(
        raw_rows=raw_rows,
        deduped_rows=int(frame_ids.shape[0]),
        duplicate_rows=duplicate_rows,
        identical_duplicate_rows=identical_duplicates,
        conflicting_duplicate_rows=conflicting_duplicates,
    )
    return PoseSequence(
        timestamps=timestamps,
        positions=positions,
        quaternions_xyzw=quaternions,
        frame_ids=frame_ids,
    ), summary


def load_init_pose(csv_path: Path, run_name: str) -> InitPose:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0] != run_name:
                continue
            return InitPose(
                run_name=row[0],
                floorplan_name=row[1],
                timestamp=float(row[2]),
                position=np.asarray([float(row[3]), float(row[4]), float(row[5])], dtype=np.float64),
                quaternion_xyzw=np.asarray([float(row[6]), float(row[7]), float(row[8]), float(row[9])], dtype=np.float64),
            )
    raise RuntimeError(f"Run {run_name!r} not found in {csv_path}")


def interpolate_pose_sequence(sequence: PoseSequence, query_timestamps: np.ndarray) -> PoseSequence:
    if sequence.count < 2:
        raise RuntimeError("At least two poses are required for interpolation")

    if np.any(query_timestamps < sequence.timestamps[0]) or np.any(query_timestamps > sequence.timestamps[-1]):
        raise ValueError("Query timestamps fall outside the source pose range")

    positions = np.column_stack(
        [np.interp(query_timestamps, sequence.timestamps, sequence.positions[:, axis]) for axis in range(3)]
    )
    rotations = Rotation.from_quat(sequence.quaternions_xyzw)
    slerp = Slerp(sequence.timestamps, rotations)
    quaternions = slerp(query_timestamps).as_quat()

    return PoseSequence(
        timestamps=np.asarray(query_timestamps, dtype=np.float64),
        positions=positions.astype(np.float64),
        quaternions_xyzw=quaternions.astype(np.float64),
    )


def pose_sequence_to_matrices(sequence: PoseSequence) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], sequence.count, axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(sequence.quaternions_xyzw).as_matrix()
    matrices[:, :3, 3] = sequence.positions
    return matrices


def matrices_to_pose_sequence(
    matrices: np.ndarray,
    timestamps: np.ndarray,
    frame_ids: np.ndarray | None = None,
) -> PoseSequence:
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected matrices with shape (N, 4, 4), found {matrices.shape}")
    positions = matrices[:, :3, 3]
    quaternions = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return PoseSequence(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        positions=positions.astype(np.float64),
        quaternions_xyzw=quaternions.astype(np.float64),
        frame_ids=None if frame_ids is None else np.asarray(frame_ids),
    )
