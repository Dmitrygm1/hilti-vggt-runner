import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

POSE_FILE = "../office_loop_run1.txt"

def load_poses(path):
    poses = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue

            frame = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])

            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])

            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = t

            poses[frame] = T

    return poses


poses = load_poses(POSE_FILE)

files = sorted(glob.glob("*.npz"), key=lambda x: float(os.path.splitext(x)[0]))

all_points = []

for f in files:
    frame_id = float(os.path.splitext(f)[0])

    if frame_id not in poses:
        continue

    data = np.load(f)

    pc = data["pointcloud"]
    mask = data["mask"].astype(bool)

    pts = pc[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) == 0:
        continue

    T = poses[frame_id]

    Rm = T[:3, :3]
    t = T[:3, 3]

    pts_world = (Rm @ pts.T).T + t

    all_points.append(pts_world)

points = np.vstack(all_points)

print("Merged raw points:", points.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd = pcd.voxel_down_sample(voxel_size=0.02)

pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

print("After cleanup:", np.asarray(pcd.points).shape)

o3d.io.write_point_cloud(
    "office_loop_global_reconstruction.ply",
    pcd
)

print("Saved office_loop_global_reconstruction.ply")