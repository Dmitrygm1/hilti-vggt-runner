# hilti-vggt-runner

Reproducible runner and glue code for preparing Hilti inputs, running `VGGT-SLAM`, exporting a viewable `.ply` reconstruction, and evaluating trajectories and floor-plan consistency.

## What This Repo Owns

- Path and sequence configuration.
- Source preparation for either:
  - stitched equirectangular MP4 input
  - raw Hilti ROS2 bag input stitched into equirectangular frames
- `VGGT-SLAM` command orchestration.
- Export of logged framewise point clouds to a global `.ply`.
- Evaluation of estimated trajectories against GT and floor-plan consistency checks.
- Lightweight tests for config, extraction, and export logic.

Method-specific code stays in `/home/drudshin/projects/VGGT-SLAM`. The Hilti challenge reference repo stays in `/home/drudshin/projects/hilti-trimble-slam-challenge-2026`.

## Prerequisites

- Python environment: `/home/drudshin/jupyter-vggt`
- `VGGT-SLAM`: `/home/drudshin/projects/VGGT-SLAM`
- Hilti challenge repo: `/home/drudshin/projects/hilti-trimble-slam-challenge-2026`
- Local path config: [`configs/local_paths.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/local_paths.yaml)

Activate the environment before running scripts:

```bash
source ~/jupyter-vggt/bin/activate
cd ~/projects/hilti-vggt-runner
```

## GPU Shell

Preparation and export can run on a login node. `VGGT-SLAM` itself should be launched from a GPU shell.

```bash
srun --pty -A 3dv --gpus=5060ti:1 -t 120 bash --login
```

Important:

- `srun --pty ... bash --login` starts a new shell on the GPU node.
- Wait for the prompt to change before running the rest of your commands.
- If you paste a whole block before the GPU prompt appears, the later commands may never run in the new shell.

After the GPU prompt appears:

```bash
source ~/jupyter-vggt/bin/activate
export TORCH_HOME=/work/scratch/$USER/torch-cache
cd ~/projects/hilti-vggt-runner
```

## Configs

- [`configs/paths.example.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/paths.example.yaml): template for path setup
- [`configs/local_paths.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/local_paths.yaml): repo-local default using `${USER}`
- [`configs/hilti_floor_ug1_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1.yaml): MP4 baseline path
- [`configs/hilti_floor1_2025-05-05_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor1_2025-05-05_run1.yaml): GT-backed MP4 reference run
- [`configs/hilti_floor_ug1_run1_rosbag.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_rosbag.yaml): rosbag parity run at `stride=10`
- [`configs/hilti_floor_ug1_run1_rosbag_dense.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_rosbag_dense.yaml): denser rosbag run at `stride=6`
- [`configs/eval_floor1_2025-05-05_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/eval_floor1_2025-05-05_run1.yaml): supervisor-facing GT evaluation for floor 1
- [`configs/eval_floor_ug1_2025-06-18_run1_rosbag_dense.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/eval_floor_ug1_2025-06-18_run1_rosbag_dense.yaml): supervisor-facing floor-plan evaluation for June 18 dense rosbag

Sequence configs now use an `input.type` discriminator:

- `mp4`: `source_mp4`, `sample_fps`, `rotate_180`
- `rosbag`: `rosbag_db3`, `calibration_yaml`, `mask0`, `mask1`, `sphere_m`, `stride`, `max_frames`, `rotate_180`, `sync_tolerance_ns`

## MP4 Workflow

Prepare:

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

Run VGGT:

```bash
python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

Export:

```bash
python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

Use `--profile full` for the full MP4 run.

## GT-Backed Reference Run

Use the floor 1 MP4 when you need a reconstruction that can be compared against a released full GT trajectory.

Prepare:

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1.yaml \
  --profile full
```

Run VGGT:

```bash
python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1.yaml \
  --profile full
```

Export:

```bash
python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1.yaml \
  --profile full
```

## Rosbag Workflow

The rosbag path is meant to test whether calibrated dual-fisheye stitching works better than the vendor MP4 for Hilti UG1.

### Rosbag Prerequisite

Stage the exact bag for the run at:

```text
${HOME}/data/hilti-2026/floor_UG1/2025-06-18/run_1/rosbag/rosbag.db3
```

That matches the default config in [`configs/hilti_floor_ug1_run1_rosbag.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_rosbag.yaml). If the bag is somewhere else, update that config.

The runner uses the challenge repo’s calibration and masks:

- `config/hilti_openvins/kalibr_imucam_chain.yaml`
- `config/hilti_openvins/mask_cam0.png`
- `config/hilti_openvins/mask_cam1.png`

### Smoke

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag.yaml \
  --profile smoke

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag.yaml \
  --profile smoke

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag.yaml \
  --profile smoke
```

### Full Parity Run

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag.yaml \
  --profile full

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag.yaml \
  --profile full

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag.yaml \
  --profile full
```

### Denser Follow-Up

If the parity run stitches cleanly but still underuses the scene, rerun with the dense config:

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag_dense.yaml \
  --profile full
```

Important:

- Rosbag stitching applies a 180° rotation by default because the Hilti raw camera images are inverted.
- Smoke preparation creates a partial frame cache. Run `prepare_hilti_data.py --profile full` before a full reconstruction or the launcher will stop with a clear error.
- Bag timestamps are written to `frame_manifest.csv`, not encoded into filenames. This avoids the `VGGT-SLAM` filename parsing trap where only the first numeric token is used as the frame id.

## Standalone Rosbag Stitcher

If you want to stitch frames without using the full runner:

```bash
python scripts/preprocessing/extract_hilti_frames.py \
  --bag ~/data/hilti-2026/floor_UG1/2025-06-18/run_1/rosbag/rosbag.db3 \
  --yaml ~/projects/hilti-trimble-slam-challenge-2026/config/hilti_openvins/kalibr_imucam_chain.yaml \
  --mask0 ~/projects/hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam0.png \
  --mask1 ~/projects/hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam1.png \
  --out_dir /work/scratch/$USER/hilti-vggt/manual_stitch/floor_UG1_run_1/frames \
  --stride 10 \
  --rotate_180
```

This wrapper writes:

- stitched frames under the requested `--out_dir`
- `frame_manifest.csv` next to the frame folder
- `stitch_summary.yaml` next to the frame folder
- `frame_preview.jpg` next to the frame folder

## Outputs

Outputs live under:

```text
/work/scratch/$USER/hilti-vggt/runs/
```

`/work/scratch` is the ETH cluster scratch filesystem, not your home directory. The actual path is still user-specific, for example:

```text
/work/scratch/drudshin/hilti-vggt/...
```

This is intentional because extracted frames, dense logs, and reconstructions are too large for the home quota.

Typical run layout:

- `frames/`: canonical prepared frames
- `smoke_frames/`: symlinked smoke subset
- `frame_manifest.csv`: per-frame source metadata
- `source_metadata.yaml`: generic source prep summary
- `stitch_summary.yaml`: rosbag-specific stitch summary
- `frame_preview.jpg`: contact sheet from extracted frames
- `smoke/vggt/poses.txt`: VGGT pose log
- `smoke/vggt/poses_logs/*.npz`: framewise world-frame point clouds from VGGT
- `smoke/exports/*.ply`: smoke export
- `full/exports/*.ply`: full export

## Evaluation

The evaluator reads the actual reconstruction artifacts from a run’s `resolved_config.yaml`. It does not guess where `poses.txt`, `frame_manifest.csv`, or the dense logs live.

Important artifact semantics:

- `VGGT-SLAM` writes `poses.txt` as `frame_id tx ty tz qx qy qz qw`
- `frame_id` is not a timestamp
- `frame_manifest.csv` is the bridge from `frame_id` to time
- rosbag manifests are already absolute timestamps
- MP4 manifests are video-relative timestamps, so GT evaluation needs a configured absolute start time

### Full Evaluation

```bash
python scripts/evaluation/run_full_evaluation.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1/full/resolved_config.yaml \
  --eval-config configs/eval_floor1_2025-05-05_run1.yaml
```

### Trajectory Only

```bash
python scripts/evaluation/evaluate_trajectory.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1/full/resolved_config.yaml \
  --eval-config configs/eval_floor1_2025-05-05_run1.yaml
```

### Floorplan Only

```bash
python scripts/evaluation/evaluate_floorplan.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1_rosbag_dense/full/resolved_config.yaml \
  --eval-config configs/eval_floor_ug1_2025-06-18_run1_rosbag_dense.yaml
```

### Evaluation Outputs

Evaluation outputs live next to the run under:

```text
<profile_root>/evaluation/<eval_name>/
```

Typical files:

- `metrics.json`: machine-readable nested metrics
- `metrics.csv`: flattened one-row summary
- `matched_poses.csv`: aligned estimate vs GT samples for trajectory runs
- `report.md`: supervisor-facing summary with linked plots
- `plots/trajectory_xy_best_fit.png`
- `plots/translation_error_vs_time.png`
- `plots/translation_error_histogram.png`
- `plots/rpe_vs_time.png`
- `plots/floorplan_overlay.png`
- `plots/wall_consistency_overlay.png`

### Current Evaluation Scope

- `floor_1_2025-05-05_run_1`: full GT trajectory evaluation plus floor-plan consistency
- `floor_UG1_2025-06-18_run_1_rosbag_dense`: init-pose anchor plus floor-plan consistency only

The June 18 run has an init pose in `init_gt_poses.csv`, but there is no released full GT trajectory txt for it in the checked-out challenge repo.

### Evaluation Caveats

- The evaluator does not use the official challenge coverage score as a headline metric because VGGT outputs are sparse by design after frame sampling.
- MP4 GT evaluation assumes `absolute_start_time_seconds + frame_manifest timestamp_seconds`; for the Hilti reference videos this is configured as `10000.0` seconds based on the challenge README.
- Floor-plan consistency is an as-planned consistency check, not authoritative as-built truth.

## Viewing The Point Cloud

Do not expect `open3d.visualization.draw_geometries(...)` to work on the compute node. The GPU nodes are headless and usually do not have an interactive OpenGL display.

For a headless cluster-side sanity check:

```bash
python scripts/inspect_pointcloud.py \
  /work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1_rosbag/full/exports/floor_UG1_2025-06-18_run_1_rosbag.ply \
  --preview-path /work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1_rosbag/full/exports/floor_UG1_2025-06-18_run_1_rosbag_preview.png
```

That prints point count and bounds, and writes a static preview image. For interactive viewing, copy the `.ply` locally and open it in CloudCompare or MeshLab.

## Notes

- The office-loop baseline worked end to end, so VGGT itself and the `.npz` to `.ply` path are known-good.
- The Hilti MP4 run was technically successful but visually poor, which is why the rosbag path exists.
- The challenge README warns that any dual-fisheye to single equirect panorama is only approximate because the two cameras do not share one optical center.
- The exporter uses the logged `.npz` files directly and does not reapply poses, because `VGGT-SLAM` already writes those point clouds in world coordinates.
