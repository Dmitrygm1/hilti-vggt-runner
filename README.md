# hilti-vggt-runner

Reproducible runner and glue code for taking a Hilti equirectangular walkthrough video, preparing it for `VGGT-SLAM`, and exporting a viewable point cloud reconstruction.

## What This Repo Owns

- Path and sequence configuration.
- MP4 frame extraction for the upside-down Hilti equirectangular video.
- `VGGT-SLAM` command orchestration.
- Export of logged framewise point clouds to a standard `.ply`.
- Lightweight tests for config, extraction, and export logic.

Method-specific code stays in `/home/drudshin/projects/VGGT-SLAM`. The Hilti challenge reference repo stays untouched in `/home/drudshin/projects/hilti-trimble-slam-challenge-2026`.

## Prerequisites

- Existing Python environment: `/home/drudshin/jupyter-vggt`
- Existing `VGGT-SLAM` checkout: `/home/drudshin/projects/VGGT-SLAM`
- Existing Hilti challenge repo: `/home/drudshin/projects/hilti-trimble-slam-challenge-2026`
- Source MP4:
  `/home/drudshin/data/hilti-2026/raw/floor_UG1/2025-06-18/run_1/floor_UG1_2025-06-18_run_1.mp4`

Activate the environment before running any script:

```bash
source ~/jupyter-vggt/bin/activate
cd ~/projects/hilti-vggt-runner
```

## GPU Shell

Preparation and export can run without a GPU. `VGGT-SLAM` itself should be run from a GPU shell.

Example:

```bash
srun --pty -A 3dv --gpus=5060ti:1 -t 120 bash --login
```

Important:

- `srun --pty ... bash --login` starts a new shell on the GPU node.
- Wait until the prompt changes to something like `drudshin@studgpu-node01:...$`.
- Only then run the rest of the commands.
- If you paste the whole block at once, the lines after `srun` may not execute in the new shell.

After the GPU prompt appears:

```bash
source ~/jupyter-vggt/bin/activate
export TORCH_HOME=/work/scratch/$USER/torch-cache
cd ~/projects/hilti-vggt-runner
```

If you prefer a single command, this form is more reliable than pasting a full block before the GPU prompt:

```bash
srun --pty -A 3dv --gpus=5060ti:1 -t 120 bash --login -lc '
source ~/jupyter-vggt/bin/activate
export TORCH_HOME=/work/scratch/$USER/torch-cache
cd ~/projects/hilti-vggt-runner
python scripts/prepare_hilti_data.py --paths configs/local_paths.yaml --sequence configs/hilti_floor_ug1_run1.yaml --profile full
python scripts/run_vggt_on_sequence.py --paths configs/local_paths.yaml --sequence configs/hilti_floor_ug1_run1.yaml --profile full
python scripts/export_results.py --paths configs/local_paths.yaml --sequence configs/hilti_floor_ug1_run1.yaml --profile full
'
```

## Configs

- Copy or adapt [`configs/paths.example.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/paths.example.yaml) for your machine.
- A repo-local default [`configs/local_paths.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/local_paths.yaml) is included and uses environment-variable expansion instead of a hardcoded username.
- The single-floor experiment config is [`configs/hilti_floor_ug1_run1.yaml`](/home/drudshin/projects/hilti-vggt-runner/configs/hilti_floor_ug1_run1.yaml).

## Single-Floor Workflow

1. Prepare the upside-down equirectangular MP4 into a canonical frame directory in scratch.

```bash
python scripts/prepare_hilti_data.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

2. Run `VGGT-SLAM` headlessly on the chosen profile.

```bash
python scripts/run_vggt_on_sequence.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

3. Export the logged framewise point clouds to `.ply`.

```bash
python scripts/export_results.py \
  --paths configs/local_paths.yaml \
  --sequence configs/hilti_floor_ug1_run1.yaml \
  --profile smoke
```

For the full reconstruction, rerun the same commands with `--profile full`.

## Outputs

Outputs live under:

```text
/work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1/
```

`/work/scratch` is the ETH cluster scratch filesystem, not your home directory. The actual path is still user-specific, for example:

```text
/work/scratch/drudshin/hilti-vggt/...
```

This is intentional because the extracted frames, dense logs, and reconstruction artifacts are too large for a 20 GB home quota.

Important paths:

- `frames/`: canonical extracted frames from the source MP4
- `smoke_frames/`: symlinked subset for quick validation
- `smoke/vggt/poses.txt`: VGGT pose log
- `smoke/vggt/poses_logs/*.npz`: framewise world-frame point clouds logged by VGGT
- `smoke/exports/floor_UG1_2025-06-18_run_1_smoke.ply`: exported smoke point cloud
- `full/exports/floor_UG1_2025-06-18_run_1.ply`: exported full point cloud

## Viewing The Point Cloud

Do not expect `open3d.visualization.draw_geometries(...)` to work on the cluster compute node. Those nodes are headless and usually do not have an interactive OpenGL display.

For a headless cluster-side sanity check:

```bash
python scripts/inspect_pointcloud.py \
  /work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1/full/exports/floor_UG1_2025-06-18_run_1.ply \
  --preview-path /work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1/full/exports/floor_UG1_2025-06-18_run_1_preview.png
```

That prints point-count and bounds, and writes a static preview image.

If you are on a machine with a real display and OpenGL, Open3D works:

```bash
python - <<'PY'
import open3d as o3d
pcd = o3d.io.read_point_cloud("/work/scratch/$USER/hilti-vggt/runs/floor_UG1_2025-06-18_run_1/full/exports/floor_UG1_2025-06-18_run_1.ply")
o3d.visualization.draw_geometries([pcd])
PY
```

You can also copy the `.ply` locally and open it in CloudCompare or MeshLab.

## Notes

- The input MP4 is already equirectangular, so the bag-based stitching helper in `scripts/preprocessing/extract_hilti_frames.py` is not used in the primary path.
- The video is upside down, so frame extraction applies a 180° rotation by default for this sequence.
- The exporter uses the logged `.npz` files directly and does not reapply poses, because `VGGT-SLAM` already writes those point clouds in world coordinates.
