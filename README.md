# hilti-vggt-runner

Reproducible runner, export, and evaluation code for the Hilti + `VGGT-SLAM` workflow.

This repo owns the practical glue:

- preparing Hilti inputs from either MP4 or raw ROS2 bags
- launching `VGGT-SLAM` reproducibly
- exporting a viewable global `.ply`
- evaluating trajectories and floor-plan consistency
- writing presentation-friendly reports, plots, and metrics files

Method-specific SLAM code stays in `/home/drudshin/projects/VGGT-SLAM`. The challenge reference repo stays in `/home/drudshin/projects/hilti-trimble-slam-challenge-2026`.

## What This Repo Owns

This repo is the orchestration layer. It is the place to add:

- path and run configs
- data preparation and stitching logic
- launch scripts
- reconstruction export helpers
- evaluation code and reports
- lightweight tests around parsing, config, association, and metric logic

This repo is not the place for broad method changes inside `VGGT-SLAM`.

## Environment

Expected environment:

- Python venv: `/home/drudshin/jupyter-vggt`
- `VGGT-SLAM`: `/home/drudshin/projects/VGGT-SLAM`
- Hilti challenge repo: `/home/drudshin/projects/hilti-trimble-slam-challenge-2026`
- runner repo: `/home/drudshin/projects/hilti-vggt-runner`

Activate first:

```bash
source ~/jupyter-vggt/bin/activate
cd ~/projects/hilti-vggt-runner
```

Run tests:

```bash
python -m pytest tests
```

## GPU Shell

Preparation, export, and evaluation can run on a login node. `VGGT-SLAM` itself should be run from a GPU shell.

```bash
srun --pty -A 3dv --gpus=5060ti:1 -t 120 bash --login
```

Important:

- `srun --pty ... bash --login` starts a new shell on the compute node.
- Wait for the prompt to change before running the next commands.
- If you paste the whole block before the GPU prompt appears, the later commands may stay behind in the original shell and never execute where you expect.

After the GPU prompt appears:

```bash
source ~/jupyter-vggt/bin/activate
export TORCH_HOME=/work/scratch/$USER/torch-cache
cd ~/projects/hilti-vggt-runner
```

## Configs

Core configs:

- [`configs/paths.example.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/paths.example.yaml): template path config
- [`configs/local_paths.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/local_paths.yaml): local default path config

Reconstruction configs:

- [`configs/hilti_floor1_2025-05-05_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor1_2025-05-05_run1.yaml): GT-backed floor 1 MP4 reference run
- [`configs/hilti_floor1_2025-05-05_run1_vggt_front_level_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor1_2025-05-05_run1_vggt_front_level_300.yaml): Floor 1 rosbag, IMU-leveled front pinhole baseline
- [`configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_frame_major_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_frame_major_300.yaml): Floor 1 rosbag, IMU-leveled 4-view pinhole sequence in `frame0_v0, frame0_v1, ...` order
- [`configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml): Floor 1 rosbag, same 4-view sequence reordered into `all_v0, all_v1, ...`
- [`configs/hilti_floor_ug1_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1.yaml): UG1 MP4 debug baseline without full GT
- [`configs/hilti_floor_ug1_run1_rosbag.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_rosbag.yaml): UG1 rosbag parity/debug run without full GT
- [`configs/hilti_floor_ug1_run1_rosbag_dense.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_rosbag_dense.yaml): denser UG1 rosbag debug run without full GT
- [`configs/hilti_floor_ug1_run1_vggt_front_fixed_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_vggt_front_fixed_300.yaml): UG1 fixed front pinhole debug baseline
- [`configs/hilti_floor_ug1_run1_vggt_front_level_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_vggt_front_level_300.yaml): UG1 IMU-leveled front pinhole debug baseline
- [`configs/hilti_floor_ug1_run1_vggt_yaw4_frame_major_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_vggt_yaw4_frame_major_300.yaml): UG1 IMU-leveled 4-view debug run in `frame0_v0, frame0_v1, ...` order
- [`configs/hilti_floor_ug1_run1_vggt_yaw4_view_major_300.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1_vggt_yaw4_view_major_300.yaml): UG1 4-view debug run reordered into `all_v0, all_v1, ...`

Evaluation configs:

- [`configs/eval_floor1_2025-05-05_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/eval_floor1_2025-05-05_run1.yaml): full GT trajectory + floor-plan evaluation for floor 1
- [`configs/eval_floor_ug1_2025-06-18_run1_rosbag_dense.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/eval_floor_ug1_2025-06-18_run1_rosbag_dense.yaml): init-pose and floor-plan debug evaluation for June 18 UG1, marked `no_full_gt: true`
- [`configs/eval_floor_ug1_2025-06-18_multiview.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/eval_floor_ug1_2025-06-18_multiview.yaml): init-pose and floor-plan debug evaluation for UG1 multiview runs, marked `no_full_gt: true`

Sequence configs use `input.type`:

- `mp4`: `source_mp4`, `sample_fps`, `rotate_180`
- `rosbag`: `rosbag_db3`, `calibration_yaml`, `mask0`, `mask1`, `sphere_m`, `stride`, `max_frames`, `rotate_180`, `sync_tolerance_ns`

Sequence configs can also define a `views` block:

- `mode: equirect | pinhole_fixed | pinhole_level_imu | pinhole_level_yaw_imu`
- `width`, `height`, `fov_deg` for the rendered pinhole images
- `yaw_deg`, `pitch_deg`, `roll_deg` for fixed single-view pinholes
- `yaws_deg` and `ordering: frame_major | view_major` for multiview yaw runs
- `imu_tau`, `time_offset_ns`, `use_yaml_timeshift` for IMU leveling
- `max_physical_frames` to cap a full run without changing the raw source cache
- `evaluation_view_index` to mark which view should be used for one-pose-per-physical-frame evaluation

## Quick Start

The reconstruction workflow is always the same three steps:

1. Prepare frames
2. Run `VGGT-SLAM`
3. Export the logged framewise point clouds to `.ply`

Primary full-GT experiment, using the DA3-style Floor 1 rosbag yaw4 view-major config:

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml \
  --profile full

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml \
  --profile full

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml \
  --profile full

python scripts/evaluation/prepare_multiview_eval_inputs.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/resolved_config.yaml
```

Then evaluate:

```bash
python scripts/evaluation/run_full_evaluation.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/evaluation_inputs/view_00/resolved_config.yaml \
  --eval-config configs/eval_floor1_2025-05-05_run1.yaml
```

## Reconstruction Workflows

### MP4 Workflow

Prepare:

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

Run:

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

Use `--profile full` for the full sequence.

### GT-Backed Reference Run

Use the floor 1 MP4 when you need a reconstruction that can be compared against a released full GT trajectory.

Prepare:

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1.yaml \
  --profile full
```

Run:

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

### Rosbag Workflow

The rosbag path exists because the vendor MP4 and the stitched raw bag are meaningfully different inputs. It lets us test whether calibrated dual-fisheye stitching gives `VGGT-SLAM` a better chance than the delivered equirectangular video.

The primary Floor 1 rosbag configs expect:

```text
/work/scratch/$USER/hilti-2026/raw/floor_1/2025-05-05/run_1/rosbag/rosbag.db3
```

The UG1 rosbag configs still exist, but they are debug/no-full-GT baselines. They should not be used for supervisor-facing trajectory metrics unless a matching full GT trajectory is added later.

The runner uses the challenge repo calibration and masks:

- `config/hilti_openvins/kalibr_imucam_chain.yaml`
- `config/hilti_openvins/mask_cam0.png`
- `config/hilti_openvins/mask_cam1.png`

#### Smoke

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

#### Full Parity Run

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

#### Denser Follow-Up

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag_dense.yaml \
  --profile full

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag_dense.yaml \
  --profile full

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1_rosbag_dense.yaml \
  --profile full
```

Rosbag-specific notes:

- Stitching rotates the panorama by 180 degrees by default because the Hilti raw sensors are inverted.
- Smoke prep writes only a partial frame cache. Run full prep before a full reconstruction.
- Timestamps are written to `frame_manifest.csv`, not embedded in filenames. This avoids the `VGGT-SLAM` filename parsing trap where only the first numeric token becomes the frame id.

### VGGT Multiview Hilti Workflow

This is the DA3-inspired Hilti path for `VGGT-SLAM`:

```text
rosbag.db3 -> stitched equirect panoramas -> pinhole views -> VGGT-SLAM -> export -> evaluation
```

It exists because the raw equirectangular Hilti runs collapsed, while the teammate DA3 pipeline improved materially after switching to front-view and then yaw4 pinhole projections.

Important implementation details:

- the stitched equirect source cache is shared across the front-view and yaw4 variants
- the rendered pinhole cache is also shared across smoke and full profiles
- emitted JPEGs are always named `frame_000001.jpg`, `frame_000002.jpg`, and so on, even for multiview runs
- per-view timestamps and physical-frame grouping live in `frame_manifest.csv`
- `VGGT-SLAM` is launched with `--disable_flow_keyframes` for these curated pinhole sequences so it consumes the prepared ordering exactly as rendered

Recommended sequence order:

1. `front_level`
2. `yaw4_frame_major`
3. `yaw4_view_major`

#### Front Level Smoke

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_front_level_300.yaml \
  --profile smoke

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_front_level_300.yaml \
  --profile smoke

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_front_level_300.yaml \
  --profile smoke
```

#### Yaw4 Frame-Major Full

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_frame_major_300.yaml \
  --profile full

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_frame_major_300.yaml \
  --profile full

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_frame_major_300.yaml \
  --profile full
```

#### Yaw4 View-Major Full

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml \
  --profile full

python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml \
  --profile full

python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor1_2025-05-05_run1_vggt_yaw4_view_major_300.yaml \
  --profile full
```

Use the UG1 multiview configs only for qualitative/no-full-GT debugging.

### Standalone Rosbag Stitcher

If you want stitched panoramas without the full runner:

```bash
python scripts/preprocessing/extract_hilti_frames.py \
  --bag /work/scratch/$USER/hilti-2026/raw/floor_UG1/2025-06-18/run_1/rosbag/rosbag.db3 \
  --yaml ~/projects/hilti-trimble-slam-challenge-2026/config/hilti_openvins/kalibr_imucam_chain.yaml \
  --mask0 ~/projects/hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam0.png \
  --mask1 ~/projects/hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam1.png \
  --out_dir /work/scratch/$USER/hilti-vggt/manual_stitch/floor_UG1_run_1/frames \
  --stride 10 \
  --rotate_180
```

This writes:

- stitched frames
- `frame_manifest.csv`
- `stitch_summary.yaml`
- `frame_preview.jpg`

## Output Layout

Heavy outputs live under:

```text
/work/scratch/$USER/hilti-vggt/runs/
```

This is the ETH scratch filesystem, not your home directory. That is intentional because frames, logs, and point clouds are too large for the home quota.

Typical run layout:

```text
<outputs_root>/
  prepared/
    <source_cache>/
      frames/
      frame_manifest.csv
      source_metadata.yaml
      stitch_summary.yaml
      frame_preview.jpg
      views/
        <view_cache>/
          frames/
          frame_manifest.csv
          view_metadata.yaml
          frame_preview.jpg
  runs/
    <run_name>/
      smoke_frames/
      smoke/
        resolved_config.yaml
        logs/
        vggt/
          poses.txt
          poses_logs/*.npz
        exports/*.ply
      full/
        resolved_config.yaml
        logs/
        vggt/
          poses.txt
          poses_logs/*.npz
        exports/*.ply
        evaluation_inputs/
          view_00/
            resolved_config.yaml
            poses.txt
            frame_manifest.csv
            selection.yaml
        evaluation/<eval_name>/
          report.md
          metrics.json
          metrics.csv
          matched_poses.csv
          plots/*.png
          aligned_pointclouds/
            aligned_init_anchor.ply
            aligned_rigid_se3.ply
            aligned_sim3_diagnostic.ply
```

## Evaluation

The evaluator reads the actual reconstruction artifacts from a run's `resolved_config.yaml`. It does not guess where files live.

Important artifact semantics:

- `VGGT-SLAM` writes `poses.txt` as `frame_id tx ty tz qx qy qz qw`
- `frame_id` is not a timestamp
- `frame_manifest.csv` is the bridge from `frame_id` to time
- rosbag manifests already contain absolute timestamps
- MP4 manifests contain video-relative timestamps, so GT evaluation needs a configured absolute start time

### Full Evaluation

```bash
python scripts/evaluation/run_full_evaluation.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/evaluation_inputs/view_00/resolved_config.yaml \
  --eval-config configs/eval_floor1_2025-05-05_run1.yaml
```

### Trajectory Only

```bash
python scripts/evaluation/evaluate_trajectory.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/evaluation_inputs/view_00/resolved_config.yaml \
  --eval-config configs/eval_floor1_2025-05-05_run1.yaml
```

### Floorplan Only

```bash
python scripts/evaluation/evaluate_floorplan.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1_rosbag_dense/full/resolved_config.yaml \
  --eval-config configs/eval_floor_ug1_2025-06-18_run1_rosbag_dense.yaml
```

### Multiview Evaluation

For the yaw4 runs, do not evaluate the raw multiview manifest directly. The evaluator expects one pose stream per physical frame, so derive a filtered evaluation input first:

```bash
python scripts/evaluation/prepare_multiview_eval_inputs.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/resolved_config.yaml
```

That writes a derived config under:

```text
/work/scratch/$USER/hilti-vggt/runs/<run_name>/full/evaluation_inputs/view_00/
```

Then evaluate using the derived config:

```bash
python scripts/evaluation/run_full_evaluation.py \
  --resolved-config /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/evaluation_inputs/view_00/resolved_config.yaml \
  --eval-config configs/eval_floor1_2025-05-05_run1.yaml
```

### Evaluation Outputs

Evaluation outputs live under:

```text
<profile_root>/evaluation/<eval_name>/
```

Typical files:

- `report.md`: concise supervisor-facing summary
- `metrics.json`: machine-readable nested metrics
- `metrics.csv`: flattened one-row summary
- `matched_poses.csv`: aligned estimate-vs-GT samples for trajectory runs
- `plots/trajectory_xy_best_fit.png`
- `plots/translation_error_3d_vs_time.png`
- `plots/translation_error_xy_vs_time.png`
- `plots/translation_error_3d_histogram.png`
- `plots/translation_error_xy_histogram.png`
- `plots/rpe_translation_m_vs_index.png`
- `plots/rpe_rotation_deg_vs_index.png`
- `plots/floorplan_overlay.png`
- `plots/pointcloud_floorplan_overlay.png`
- `plots/wall_consistency_overlay.png`
- `aligned_pointclouds/aligned_init_anchor.ply`
- `aligned_pointclouds/aligned_rigid_se3.ply`
- `aligned_pointclouds/aligned_sim3_diagnostic.ply`

The exported reconstruction PLY under `exports/` is in the raw VGGT reconstruction frame. The `aligned_pointclouds/aligned_*.ply` files are copies transformed into the same map/GT frames used by the named evaluation modes. Use those aligned PLYs when comparing against floorplans or trajectory metrics; use the raw export for method-debug CloudCompare inspection.

### Current Evaluation Scope

- `floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300`: primary full-GT trajectory evaluation plus floor-plan consistency
- `floor_1_2025-05-05_run_1`: MP4 fallback baseline with full GT trajectory evaluation plus floor-plan consistency
- `floor_UG1_2025-06-18_run_1_rosbag_dense`: init-pose anchor plus floor-plan consistency only

The June 18 run has an init pose in `init_gt_poses.csv`, but there is no released full GT trajectory txt for it in the checked-out challenge repo.

## How To Read The Metrics

This section is the intended interpretation guide for the numbers in `metrics.json`, `metrics.csv`, and `report.md`.

### 1. Run-Health Metrics

These are not "accuracy" metrics. They tell you whether the output even looks structurally plausible before you start comparing it to ground truth.

| Metric | What it means | What it helps you detect |
| --- | --- | --- |
| `estimated_pose_summary.raw_rows` | Raw number of rows read from `poses.txt`. | Logging anomalies or duplicate entries. |
| `estimated_pose_summary.deduped_rows` | Unique poses after deduplicating repeated frame ids. | How much usable trajectory was actually produced. |
| `estimated_pose_summary.duplicate_rows` | Number of rows removed because the same frame id appeared again. | Repeated logging or overwrites. |
| `estimated_sequence_health.pose_count` | Final count of usable estimated poses. | Whether the run produced enough samples to evaluate. |
| `estimated_sequence_health.path_length_m` | Total path length implied by the estimated trajectory. | Collapse: if a building-scale walk yields a sub-meter path, the estimate likely collapsed. |
| `estimated_sequence_health.xy_extent_m` | Planar spread of the trajectory. | Whether the reconstruction occupies a meaningful area in the floor plane. |
| `estimated_sequence_health.xyz_extent_m` | Axis-aligned extent of the estimate. | Whether the map is spatially spread or compressed into a tiny blob. |

Use these first. If the path length or XY extent is tiny, the method failed before any sophisticated metric matters.

### 2. Alignment Modes

Trajectory errors are always reported under an explicit alignment mode. They answer different questions.

| Alignment mode | What it does | What the resulting errors are meant to show |
| --- | --- | --- |
| `rigid_se3` | Fits one best rigid transform between the estimated and GT trajectories. No scale change. | "If I place the whole estimated trajectory into the GT frame as favorably as possible, how close is its shape and position sequence?" This is the best headline metric for post-hoc trajectory agreement. |
| `init_anchor` | Aligns using the known init pose only, with yaw + translation anchoring. | "If I only give the run one realistic map anchor, how well does it stay consistent afterward?" This is more deployment-like than best-fit alignment. |
| `sim3_diagnostic` | Fits rigid transform plus scale. | Diagnostic only. It answers "would a scale change rescue the trajectory?" It should not be used as the headline result for a metric-sensitive presentation. |

What these modes are not:

- `rigid_se3` is not a fair measure of online localization readiness, because it lets the evaluator reposition the entire estimate after the fact.
- `sim3_diagnostic` is not a claim that scale correction is acceptable. It is only there to diagnose whether scale is a dominant failure mode.
- `init_anchor` depends on having a valid init pose and a close-enough estimated pose near that time.

### 3. Trajectory Metrics

These metrics compare estimated poses against GT after the selected alignment.

#### Absolute Trajectory Error

ATE measures pointwise position disagreement between aligned estimated poses and GT at the same timestamps.

| Metric family | What it means | Why it matters |
| --- | --- | --- |
| `ate_xy_m_*` | Position error in the floor plane only. | Best metric when the practical question is "how wrong is the walk in the building layout?" |
| `ate_3d_m_*` | Full 3D position error. | Useful when vertical drift matters too, but often less important than XY for floor-based navigation. |

For each family, the suffixes mean:

| Suffix | What it shows |
| --- | --- |
| `mean` | Average error. Sensitive to broad overall bias. |
| `median` | Typical error. More robust to a few bad outliers. |
| `rmse` | Root-mean-square error. Useful as a headline because it penalizes large misses more strongly. |
| `p95` | 95th percentile. Shows the "bad but not worst-case" tail. |
| `max` | Worst matched error. Good for spotting catastrophic spikes, not as a headline metric by itself. |

#### Rotation Error

`rotation_error_deg_*` is the per-pose orientation disagreement after the same alignment used for ATE.

This is meant to show whether the method points the camera in roughly the right direction over time. It is useful, but usually secondary to translational error for a first supervisor-facing summary.

#### Relative Pose Error

RPE is a short-horizon drift metric. In this runner it is computed over a fixed time horizon, currently `1.0 s` by default.

| Metric family | What it means | What it is good for |
| --- | --- | --- |
| `rpe_translation_m_*` | Local translation drift over the chosen horizon. | Separating local consistency from global misalignment. |
| `rpe_rotation_deg_*` | Local rotation drift over the chosen horizon. | Detecting unstable short-term orientation drift. |

How to read it:

- A run can have poor ATE but decent RPE: local motion is somewhat self-consistent, but the whole trajectory is globally wrong.
- A run with poor RPE is locally unstable, which usually means the SLAM itself is not tracking well.

#### Association and Anchoring

These fields explain how the trajectory comparison was made:

| Metric | What it means |
| --- | --- |
| `matched_pose_count` | Number of estimated poses that were actually compared against GT after trimming to the overlapping time range. |
| `anchor_frame_id` | Estimated frame used for init-pose anchoring. |
| `anchor_timestamp` | Timestamp of the anchored estimated pose. |
| `anchor_timestamp_delta_seconds` | Time gap between the anchored estimate and the GT init pose. Smaller is better. |

GT handling details:

- the first `ignore_initial_seconds` are excluded before evaluation
- GT is interpolated to the estimated timestamps
- for MP4 runs, estimated absolute time is reconstructed as `absolute_start_time_seconds + frame_manifest timestamp_seconds`

### 4. Floor-Plan Metrics

These metrics are 2D consistency checks against the as-planned floor plan PNG. They are not full geometric truth metrics.

Pipeline summary:

1. Anchor the estimated run into the floor-plan frame using the init pose.
2. Project reconstruction evidence into the floor plane.
3. Keep only points inside a height band and only cells with enough vertical extent to look wall-like.
4. Restrict evaluation to a corridor around the walked trajectory.
5. Compare the resulting evidence mask against the floor-plan wall mask.

#### Evidence Metrics

| Metric | What it means | What it helps you detect |
| --- | --- | --- |
| `point_evidence_source` | Whether wall evidence came from raw `.npz` logs or the exported `.ply`. | Which source supported the metric. |
| `point_evidence_input_points` | Number of 3D points examined before raster filtering. | Whether the evidence stage had enough raw material. |
| `point_evidence_cells` | Number of grid cells that survived evidence filtering. | Whether the run produced any usable structural signal at all. |
| `evidence_cells_in_corridor` | Evidence cells inside the evaluation corridor. | Whether the walked area contains usable wall evidence. |
| `wall_cells_in_corridor` | Floor-plan wall cells inside the same corridor. | Size of the local reference region. |
| `trajectory_corridor_cells` | Total evaluated corridor area in cells. | How large the local evaluation window was. |

If `point_evidence_cells` is zero, the floor-plan metrics are mostly telling you "the reconstruction did not yield usable wall evidence after anchoring and filtering."

#### Overlap Metrics

| Metric | What it is meant to show |
| --- | --- |
| `wall_precision` | Of the reconstruction wall evidence we accepted, what fraction lies close to a planned wall? High precision means the evidence is mostly on-map rather than sprayed elsewhere. |
| `wall_recall` | Of the planned walls near the walked trajectory, what fraction is supported by reconstruction evidence? High recall means the run explains a larger part of the local building structure. |
| `wall_f1` | Balance between precision and recall. Good compact summary if you want one overlap headline number. |
| `wall_iou` | Conservative overlap score between reconstruction evidence and planned walls inside the corridor. Useful, but harsher than F1. |

#### Distance Metrics

| Metric | What it is meant to show |
| --- | --- |
| `wall_distance_mean_m` | Average distance from reconstruction evidence to the nearest floor-plan wall. |
| `wall_distance_median_m` | Typical nearest-wall distance. |
| `wall_distance_p95_m` | Near-worst-case but still robust wall mismatch. Good headline distance metric. |

Distance metrics are often easier to interpret than IoU when the evidence is sparse.

### 5. Plots

The plots are intended to complement the numbers:

| Plot | What it is meant to show |
| --- | --- |
| `trajectory_xy_best_fit.png` | Estimated and GT trajectories after best-fit rigid alignment in the floor plane. |
| `translation_error_3d_vs_time.png` | Where full 3D position error grows or spikes along the run. |
| `translation_error_xy_vs_time.png` | Where floor-plane position error grows or spikes along the run. |
| `translation_error_3d_histogram.png` | Distribution of full 3D translational error. |
| `translation_error_xy_histogram.png` | Distribution of floor-plane translational error. |
| `rpe_translation_m_vs_index.png` | Short-horizon local translation drift over the matched pose pairs. |
| `rpe_rotation_deg_vs_index.png` | Short-horizon local rotation drift over the matched pose pairs. |
| `floorplan_overlay.png` | Anchored trajectory and evidence over the floor plan. Good first visual for supervisor review. |
| `pointcloud_floorplan_overlay.png` | Projected point evidence from the same transform and filters used by the floorplan metrics. |
| `wall_consistency_overlay.png` | Which evidence agrees with the floor plan and which parts do not. |

### 6. What To Put In Front Of A Supervisor

For a short presentation, lead with:

1. Run-health diagnostics: pose count, path length, XY extent
2. `ATE XY RMSE` from `rigid_se3` for GT-backed runs
3. `RPE translation RMSE` to separate local drift from global misalignment
4. `wall_precision`, `wall_recall`, and `wall_distance_p95_m` for floor-plan consistency
5. `trajectory_xy_best_fit.png` and `floorplan_overlay.png`

For the June 18 UG1 dense run specifically, be explicit that:

- there is no released full GT trajectory in the checked-out challenge repo
- the report is therefore a localization-anchor + floor-plan consistency report, not a full GT trajectory benchmark

## Evaluation Caveats

- The evaluator does not use the official challenge coverage score as a headline metric because the VGGT pipeline outputs sparse keyframe poses by design.
- MP4 GT evaluation assumes `absolute_start_time_seconds + frame_manifest timestamp_seconds`; for the Hilti reference videos this is configured as `10000.0` seconds based on the challenge README.
- Floorplan plots are not screenshots of the CloudCompare PLY. They project point evidence after the selected evaluation transform, height band, minimum point-count filter, and vertical-extent filter.
- Floor-plan consistency is an as-planned consistency check, not authoritative as-built truth.
- Dual-fisheye to equirectangular stitching is only approximate because the two physical cameras do not share one optical center.

## Viewing The Point Cloud

Do not expect `open3d.visualization.draw_geometries(...)` to work on the compute node. The cluster nodes are headless and usually do not provide an interactive OpenGL display.

For a headless sanity check:

```bash
python scripts/inspect_pointcloud.py \
  /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/exports/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300.ply \
  --preview-path /work/scratch/$USER/hilti-vggt/runs/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300/full/exports/floor_1_2025-05-05_run_1_vggt_yaw4_view_major_300_preview.png
```

For interactive viewing, copy the `.ply` locally and open it in CloudCompare or MeshLab. Use the raw `exports/*.ply` to inspect VGGT's reconstruction frame. Use `evaluation/<eval_name>/aligned_pointclouds/aligned_*.ply` when you want the same coordinate frame used by metrics and floorplan overlays.

## Notes

- The `office_loop` baseline worked end to end, so `VGGT-SLAM` itself and the `.npz` to `.ply` path are known-good.
- Hilti runs have so far been method-limited rather than runner-limited: the orchestration, export, and evaluation all execute successfully, but the resulting trajectory and map quality are still poor.
- The exporter uses the logged `.npz` files directly and does not reapply poses, because `VGGT-SLAM` already writes those point clouds in world coordinates.
