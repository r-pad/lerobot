"""Real-world port of the AGOS canonical plug/socket point-cloud observations.

Mirrors third_party/AGOS/data_utils/plug_canonical.py and the AGOS task
(isaacgymenvs/tasks/automate/agos_insertion.py):

* the plug cloud is captured once (bottom camera, segmented), stored in the
  fingertip frame ``p_ft = R_ft^T (p_w - t_ft)``;
* every step it is moved with proprioception ``p_w = R_ft(t) p_ft + t_ft(t)``
  and rendered bottom-up in world axes, centred on its own xy mean;
* the socket cloud (multi-view wrist scan, world frame) is rendered once
  top-down, centred on its xy bounding box;
* both use the same pixel layout (image up = world -x, image right = world +y),
  depth in metres from the virtual camera, NaN where empty.
"""

import math

import numpy as np
import torch

# AGOS `env.plug_photo.canonical` defaults (AutoMateTaskAGOS.yaml).
CANONICAL_HEIGHT = 240
CANONICAL_WIDTH = 320
CANONICAL_FOV_DEG = 73.73979365244269
CANONICAL_OFFSET_M = 0.02
CANONICAL_MAX_DEPTH_M = 0.1
PLUG_POINT_RADIUS = 2
SOCKET_POINT_RADIUS = 4


def world_points_to_fingertip_frame(points_world: np.ndarray, fingertip_pose: np.ndarray) -> np.ndarray:
    """p_local = R^T (p - t) for a 4x4 world-from-fingertip pose."""
    pose = np.asarray(fingertip_pose, dtype=np.float64).reshape(4, 4)
    return ((np.asarray(points_world, dtype=np.float64)[:, :3] - pose[:3, 3]) @ pose[:3, :3]).astype(np.float32)


def fingertip_points_to_world(points_local: np.ndarray, fingertip_pose: np.ndarray) -> np.ndarray:
    """p_world = R p_local + t for a 4x4 world-from-fingertip pose."""
    pose = np.asarray(fingertip_pose, dtype=np.float64).reshape(4, 4)
    return (np.asarray(points_local, dtype=np.float64)[:, :3] @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def green_mask(
    colors_rgb: np.ndarray,
    hue_range_deg: tuple[float, float] = (70.0, 170.0),
    min_saturation: float = 0.25,
    min_value: float = 0.15,
) -> np.ndarray:
    """Real-world stand-in for the sim plug segmentation id: keep green points (RGB uint8 or [0, 1])."""
    rgb = np.asarray(colors_rgb, dtype=np.float32)
    if rgb.size == 0:
        return np.zeros((0,), dtype=bool)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = rgb.max(axis=1)
    cmin = rgb.min(axis=1)
    delta = cmax - cmin
    saturation = np.where(cmax > 1e-6, delta / np.maximum(cmax, 1e-6), 0.0)
    # Hue in degrees; only the green-dominant branch matters here.
    hue = np.zeros_like(cmax)
    safe = np.maximum(delta, 1e-6)
    is_r = (cmax == r) & (delta > 0)
    is_g = (cmax == g) & (delta > 0) & ~is_r
    is_b = (delta > 0) & ~is_r & ~is_g
    hue[is_r] = (60.0 * ((g - b)[is_r] / safe[is_r])) % 360.0
    hue[is_g] = 60.0 * ((b - r)[is_g] / safe[is_g]) + 120.0
    hue[is_b] = 60.0 * ((r - g)[is_b] / safe[is_b]) + 240.0
    lo, hi = hue_range_deg
    return (hue >= lo) & (hue <= hi) & (saturation >= min_saturation) & (cmax >= min_value)


def render_world_canonical(
    points_world,
    colors,
    look_dir: int,
    offset_m: float = CANONICAL_OFFSET_M,
    fov_deg: float = CANONICAL_FOV_DEG,
    height: int = CANONICAL_HEIGHT,
    width: int = CANONICAL_WIDTH,
    point_radius: int = PLUG_POINT_RADIUS,
    center_xy=None,
    center_mode: str = "mean",
    max_depth_m: float = CANONICAL_MAX_DEPTH_M,
) -> dict:
    """Verbatim port of AGOS ``render_world_canonical``.

    look_dir = -1: socket, camera above looking down. look_dir = +1: plug, camera below
    looking up. Both share image up = world -x, image right = world +y.
    """
    pts = torch.as_tensor(np.asarray(points_world), dtype=torch.float32)[:, :3]
    cols = torch.as_tensor(np.asarray(colors), dtype=torch.float32)
    if cols.numel() > 0 and cols.max() > 1.0:
        cols = cols / 255.0
    H, W = int(height), int(width)
    depth = torch.full((H, W), float("nan"), dtype=torch.float32)
    rgb = torch.zeros((H, W, 3), dtype=torch.float32)
    if pts.shape[0] == 0:
        return {"depth": depth.numpy(), "rgb": rgb.numpy(), "mask": np.zeros((H, W), dtype=bool),
                "center_xy": (float("nan"), float("nan")), "camera_z": float("nan"), "px_per_m": float("nan")}
    if look_dir not in (-1, 1):
        raise ValueError(f"look_dir must be -1 or +1, got {look_dir}")
    if center_xy is None:
        if center_mode == "mean":
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        elif center_mode == "bbox":
            cx = float((pts[:, 0].min() + pts[:, 0].max()) / 2)
            cy = float((pts[:, 1].min() + pts[:, 1].max()) / 2)
        else:
            raise ValueError(f"Unknown center_mode {center_mode!r}")
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])
    if look_dir == -1:
        cam_z = float(pts[:, 2].max()) + float(offset_m)
        d = cam_z - pts[:, 2]
    else:
        cam_z = float(pts[:, 2].min()) - float(offset_m)
        d = pts[:, 2] - cam_z
    f = (H / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    valid = (d > 1e-6) & (d < float(max_depth_m))
    u = W / 2.0 + f * (pts[:, 1] - cy) / d
    v = H / 2.0 - f * (pts[:, 0] - cx) / d
    ui = torch.round(u).long()
    vi = torch.round(v).long()
    valid &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, d, cols = ui[valid], vi[valid], d[valid], cols[valid]
    if ui.numel() > 0:
        r = int(point_radius)
        offsets = [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r]
        du = torch.tensor([o[0] for o in offsets])
        dv = torch.tensor([o[1] for o in offsets])
        uu = (ui.unsqueeze(1) + du.unsqueeze(0)).reshape(-1)
        vv = (vi.unsqueeze(1) + dv.unsqueeze(0)).reshape(-1)
        dd = d.unsqueeze(1).expand(-1, len(offsets)).reshape(-1)
        cc = cols.unsqueeze(1).expand(-1, len(offsets), -1).reshape(-1, 3)
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        uu, vv, dd, cc = uu[ok], vv[ok], dd[ok], cc[ok]
        order = torch.argsort(dd, stable=True)  # nearest first
        flat = (vv[order] * W + uu[order]).numpy()
        first = np.unique(flat, return_index=True)[1]  # nearest candidate per pixel
        sel = order[torch.from_numpy(first)]
        depth.view(-1)[torch.from_numpy(flat[first])] = dd[sel]
        rgb.view(-1, 3)[torch.from_numpy(flat[first])] = cc[sel]
    mask = torch.isfinite(depth)
    return {"depth": depth.numpy(), "rgb": rgb.numpy(), "mask": mask.numpy(), "center_xy": (cx, cy),
            "camera_z": cam_z, "px_per_m": f / float(offset_m)}


def render_plug_canonical(plug_points_fingertip: np.ndarray, plug_colors: np.ndarray, fingertip_pose: np.ndarray) -> dict:
    """Per-step plug view: move the stored cloud with the current fingertip pose, render bottom-up."""
    world = fingertip_points_to_world(plug_points_fingertip, fingertip_pose)
    return render_world_canonical(world, plug_colors, +1, point_radius=PLUG_POINT_RADIUS, center_mode="mean")


def render_socket_canonical(socket_points_world: np.ndarray, socket_colors: np.ndarray) -> dict:
    """One-off socket view from the fused wrist scan: top-down, bbox centre."""
    return render_world_canonical(
        socket_points_world, socket_colors, -1, point_radius=SOCKET_POINT_RADIUS, center_mode="bbox"
    )


def canonical_normalize(depth: np.ndarray, invalid_fill: float = 1.0) -> np.ndarray:
    """AGOS CanonicalNormalizer for one image: min-max over finite pixels, NaN -> invalid_fill."""
    depth = np.asarray(depth, dtype=np.float32)
    out = np.full_like(depth, float(invalid_fill))
    valid = np.isfinite(depth)
    if valid.any():
        vals = depth[valid]
        lo, hi = float(vals.min()), float(vals.max())
        out[valid] = (vals - lo) / (hi - lo) if hi - lo > 1e-6 else 0.0
    return out


def overlay_profiles(plug_view: dict, socket_view: dict) -> np.ndarray:
    """Debug overlay in AGOS colours: socket green, plug body red, plug face orange, face∩socket yellow."""
    plug_mask, socket_mask, plug_depth = plug_view["mask"], socket_view["mask"], plug_view["depth"]
    img = np.full(plug_mask.shape + (3,), 255, dtype=np.uint8)
    img[socket_mask] = (40, 150, 90)
    img[plug_mask] = (150, 40, 40)
    if plug_mask.any():
        face = plug_mask & (plug_depth <= np.nanmin(plug_depth) + 0.002)
        img[face] = (235, 60, 40)
        img[face & socket_mask] = (250, 220, 60)
    return img
