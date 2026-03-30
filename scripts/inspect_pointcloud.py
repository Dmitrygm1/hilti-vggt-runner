#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless inspection helper for a point cloud .ply.")
    parser.add_argument("pointcloud", help="Path to .ply point cloud")
    parser.add_argument("--preview-path", help="Optional output PNG preview path")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum points to plot in the preview")
    parser.add_argument("--elev", type=float, default=25.0, help="Matplotlib preview elevation")
    parser.add_argument("--azim", type=float, default=45.0, help="Matplotlib preview azimuth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pointcloud_path = Path(args.pointcloud).expanduser()
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    if len(points) == 0:
        raise RuntimeError(f"No points found in {pointcloud_path}")

    print(f"Point cloud: {pointcloud_path}")
    print(f"Points: {len(points)}")
    print(f"Min bound: {points.min(axis=0).tolist()}")
    print(f"Max bound: {points.max(axis=0).tolist()}")
    print(f"Has colors: {colors is not None and len(colors) == len(points)}")

    if args.preview_path:
        preview_path = Path(args.preview_path).expanduser()
        preview_path.parent.mkdir(parents=True, exist_ok=True)

        if len(points) > args.max_points:
            rng = np.random.default_rng(seed=0)
            sample_indices = rng.choice(len(points), size=args.max_points, replace=False)
            points_to_plot = points[sample_indices]
            colors_to_plot = colors[sample_indices] if colors is not None and len(colors) == len(points) else None
        else:
            points_to_plot = points
            colors_to_plot = colors if colors is not None and len(colors) == len(points) else None

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            points_to_plot[:, 0],
            points_to_plot[:, 1],
            points_to_plot[:, 2],
            c=colors_to_plot if colors_to_plot is not None else "tab:blue",
            s=2,
            linewidths=0,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(pointcloud_path.name)
        fig.tight_layout()
        fig.savefig(preview_path, dpi=200)
        plt.close(fig)
        print(f"Preview image: {preview_path}")


if __name__ == "__main__":
    main()
