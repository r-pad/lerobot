import argparse

import numpy as np
import torch

from lerobot.common.utils.pointcloud_rgbd import visualize_pointcloud_plotly


def load_colored_pcd(path: str) -> tuple[np.ndarray, np.ndarray]:
    pcd = torch.load(path, map_location="cpu")
    if isinstance(pcd, dict):
        pcd = pcd.get("points", pcd.get("pcd", pcd.get("arr_0")))
    if pcd is None:
        raise ValueError(f"No point cloud tensor found in {path}")
    if torch.is_tensor(pcd):
        pcd = pcd.detach().cpu().numpy()
    else:
        pcd = np.asarray(pcd)
    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError(f"Expected an Nx6 colored point cloud, got shape {pcd.shape}")
    return pcd[:, :3], pcd[:, 3:6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", default="/home/yinongh/automate/lerobot/debug_pcd.pth")
    parser.add_argument("--out", default="/home/yinongh/automate/lerobot/debug_pcd_plotly.html")
    parser.add_argument("--max-points", type=int, default=200000)
    args = parser.parse_args()

    points, colors = load_colored_pcd(args.pcd)
    print(f"Loaded {args.pcd}: points={points.shape}, colors={colors.shape}")
    print(f"XYZ min: {points.min(axis=0)}")
    print(f"XYZ max: {points.max(axis=0)}")
    print(f"RGB min: {colors.min(axis=0)}")
    print(f"RGB max: {colors.max(axis=0)}")
    min_z_idx = int(np.argmin(points[:, 2]))
    min_z_point = points[min_z_idx]
    print(f"Lowest z point: x={min_z_point[0]}, y={min_z_point[1]}, z={min_z_point[2]}")
    visualize_pointcloud_plotly(points, colors, max_points=args.max_points, save_path=args.out)


if __name__ == "__main__":
    main()
