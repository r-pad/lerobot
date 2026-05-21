#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def robust_normalize(depth: np.ndarray, min_depth: float | None, max_depth: float | None) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    finite = finite[finite > 0]
    if finite.size == 0:
        raise ValueError("Depth array has no positive finite values to visualize.")

    lo = np.percentile(finite, 1) if min_depth is None else min_depth
    hi = np.percentile(finite, 99) if max_depth is None else max_depth
    if hi <= lo:
        raise ValueError(f"Invalid visualization range: min={lo}, max={hi}")

    normalized = (depth - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)


def save_colormap(normalized: np.ndarray, output_path: Path, cmap_name: str) -> None:
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name)
        rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(output_path)
    except ImportError:
        grayscale = (normalized * 255).astype(np.uint8)
        Image.fromarray(grayscale, mode="L").save(output_path)


def visualize_file(input_path: Path, output_path: Path, min_depth: float | None, max_depth: float | None, cmap: str) -> None:
    depth = np.load(input_path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0]

    print(f"Loaded: {input_path}")
    print(f"shape={depth.shape}, dtype={depth.dtype}")
    print(f"finite_pixels={finite.size}/{depth.size}, positive_finite_pixels={positive.size}/{depth.size}")
    if finite.size:
        print(f"finite min={finite.min():.3f}, max={finite.max():.3f}, mean={finite.mean():.3f}")
    if positive.size:
        p = np.percentile(positive, [1, 5, 50, 95, 99])
        print(
            "positive percentiles "
            f"p1={p[0]:.3f}, p5={p[1]:.3f}, p50={p[2]:.3f}, p95={p[3]:.3f}, p99={p[4]:.3f}"
        )

    normalized = robust_normalize(depth, min_depth, max_depth)
    save_colormap(normalized, output_path, cmap)
    print(f"Saved visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a raw ZED depth .npy dump.")
    parser.add_argument(
        "input",
        nargs="?",
        default="outputs/debug_zed_depth",
        help="Path to a raw ZED depth .npy file or a directory containing raw_zed_depth_*.npy.",
    )
    parser.add_argument(
        "--output",
        default="outputs/debug_zed_depth/visualized",
        help="Path for one visualization image, or an output directory when input is a directory.",
    )
    parser.add_argument("--min-depth", type=float, default=None, help="Minimum depth in raw units, usually mm.")
    parser.add_argument("--max-depth", type=float, default=None, help="Maximum depth in raw units, usually mm.")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap name.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        npy_paths = sorted(input_path.glob("raw_zed_depth_*.npy"))
        if not npy_paths:
            raise FileNotFoundError(f"No raw_zed_depth_*.npy files found in {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Visualizing {len(npy_paths)} depth frames into {output_path}")
        for npy_path in npy_paths:
            frame_output = output_path / f"{npy_path.stem}.png"
            visualize_file(npy_path, frame_output, args.min_depth, args.max_depth, args.cmap)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_file(input_path, output_path, args.min_depth, args.max_depth, args.cmap)


if __name__ == "__main__":
    main()
