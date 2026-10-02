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


def canonical_views(
    plug_points_fingertip: np.ndarray,
    plug_colors: np.ndarray,
    fingertip_pose: np.ndarray,
    socket_points_world: np.ndarray,
    socket_colors: np.ndarray,
    shared_center: bool = True,
) -> dict:
    """Port of AGOS ``AutoMateTaskAGOS.canonical_views``.

    Plug cloud moved to the given fingertip pose and rendered bottom-up; socket rendered top-down.
    With ``shared_center`` the plug is rendered about the socket centre (so the overlay shows where
    the plug is relative to the hole); ``plug_own_center`` is the policy view (own xy-mean centre).
    """
    plug_world = fingertip_points_to_world(plug_points_fingertip, fingertip_pose)
    socket_view = render_socket_canonical(socket_points_world, socket_colors)
    center = socket_view["center_xy"] if shared_center else None
    plug_view = render_world_canonical(
        plug_world, plug_colors, +1, point_radius=PLUG_POINT_RADIUS, center_xy=center, center_mode="mean"
    )
    plug_own = (
        plug_view
        if not shared_center
        else render_world_canonical(plug_world, plug_colors, +1, point_radius=PLUG_POINT_RADIUS, center_mode="mean")
    )
    overlay = overlay_profiles(plug_view["mask"], socket_view["mask"], plug_view["depth"])
    return {"plug": plug_view, "plug_own_center": plug_own, "socket": socket_view, "overlay": overlay,
            "plug_points_world": plug_world, "fingertip_pose": np.asarray(fingertip_pose)}


def overlay_on_depth(plug_mask: np.ndarray, socket_depth: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Verbatim AGOS ``overlay_on_depth``: plug mask over the socket depth (grey, nearer = brighter),
    red where it covers socket material, yellow where it covers the hole / background."""
    d = np.asarray(socket_depth, dtype=np.float32)
    valid = np.isfinite(d)
    grey = np.zeros(d.shape, dtype=np.float32)
    if valid.any():
        x = d[valid]
        grey[valid] = 255.0 * (1.0 - (x - x.min()) / max(float(x.max() - x.min()), 1e-9))
    rgb = np.repeat(grey[..., None], 3, axis=-1)
    plug = np.asarray(plug_mask, dtype=bool)
    over = plug & valid
    hole = plug & ~valid
    for m, col in ((over, (220.0, 60.0, 40.0)), (hole, (240.0, 200.0, 40.0))):
        rgb[m] = (1.0 - alpha) * rgb[m] + alpha * np.asarray(col, dtype=np.float32)
    return rgb.astype(np.uint8)


def overlay_profiles(plug_mask: np.ndarray, socket_mask: np.ndarray, plug_depth=None) -> np.ndarray:
    """Verbatim AGOS ``overlay_profiles``: socket green, plug body dark red, plug face (within 2 mm
    of its min depth) red, face over socket yellow, on black."""
    h, w = socket_mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[socket_mask] = (40, 150, 90)
    if plug_depth is not None:
        finite = np.isfinite(plug_depth)
        face = finite & (plug_depth <= (np.nanmin(plug_depth) + 0.002 if finite.any() else 0))
        body = plug_mask & ~face
        img[body] = (150, 40, 40)
        img[face] = (235, 60, 40)
        both = face & socket_mask
    else:
        img[plug_mask] = (235, 60, 40)
        both = plug_mask & socket_mask
    img[both] = (250, 220, 60)
    return img


PAPER_PLUG_RGB = np.array([232.0, 128.0, 120.0])  # salmon


def paper_overlay(plug_view: dict, socket_view: dict, radius_frac: float | None = None,
                  socket_grey_max: float = 0.85) -> np.ndarray:
    """Paper-style AGOS overlay: white background, socket virtual image in grey levels by depth (top
    face black, deeper lighter, as the policy sees it), plug virtual image in salmon on top, shaded by
    depth (bottom face lightest, body darker). ``radius_frac`` optionally masks everything outside a
    centred disk to black.
    """
    h, w = socket_view["mask"].shape
    img = np.full((h, w, 3), 255.0, dtype=np.float32)
    socket_depth = np.asarray(socket_view["depth"], dtype=np.float32)
    socket_mask = np.isfinite(socket_depth)
    if socket_mask.any():
        # Same grey levels as the policy input (AGOS min-max): top face black, deeper (hole) lighter;
        # capped below white so the deepest socket pixels stay distinct from the background.
        norm = canonical_normalize(socket_depth)
        img[socket_mask] = (socket_grey_max * 255.0 * norm[socket_mask])[:, None]
    plug_mask = plug_view["mask"]
    if plug_mask.any():
        d = plug_view["depth"][plug_mask]
        t = (d - d.min()) / max(float(d.max() - d.min()), 1e-9)  # 0 = nearest (tip face)
        shade = 1.0 - 0.35 * t
        img[plug_mask] = PAPER_PLUG_RGB[None, :] * shade[:, None]
        # thin dark outline of the plug face profile for readability
        from scipy.ndimage import binary_closing, binary_fill_holes

        face = plug_mask & (np.nan_to_num(plug_view["depth"], nan=np.inf) <= d.min() + 0.002)
        face = binary_fill_holes(binary_closing(face, iterations=2))  # splat gaps are not edges
        edge = face & ~(np.roll(face, 1, 0) & np.roll(face, -1, 0) & np.roll(face, 1, 1) & np.roll(face, -1, 1))
        img[edge] = PAPER_PLUG_RGB * 0.55
    if radius_frac is not None:
        yy, xx = np.mgrid[0:h, 0:w]
        radius = radius_frac * min(h, w)
        outside = (yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2 > radius ** 2
        img[outside] = 0.0
    return img.astype(np.uint8)


def agos_figure(wrist_rgbs: list, overlays: list, path: str, num_columns: int = 6, title: str | None = None) -> str:
    """Two-row figure like the AGOS paper: (a) wrist observations, (b) AGOS overlays, evenly sampled."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n = len(overlays)
    if n == 0:
        return ""
    idx = np.unique(np.linspace(0, n - 1, min(num_columns, n)).round().astype(int))
    ref_wrist = next((w for w in wrist_rgbs if w is not None), None)
    wrist_h, wrist_w = (ref_wrist.shape[:2] if ref_wrist is not None else (180, 320))
    over_h, over_w = overlays[0].shape[:2]
    col_w = 2.0  # inches per column
    row_h = [col_w * wrist_h / wrist_w, col_w * over_h / over_w]
    label_h = 0.28
    fig = plt.figure(figsize=(col_w * len(idx), sum(row_h) + 2 * label_h + (0.3 if title else 0.0)))
    grid = fig.add_gridspec(4, len(idx), height_ratios=[label_h, row_h[0], label_h, row_h[1]],
                            hspace=0.0, wspace=0.04, left=0.005, right=0.995, top=0.93 if title else 0.995, bottom=0.005)
    for col, i in enumerate(idx):
        for row, img in ((1, wrist_rgbs[i]), (3, overlays[i])):
            ax = fig.add_subplot(grid[row, col])
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
    for row, text in ((0, "(a) Wrist Observation"),
                      (2, "(b) AGOS Overlay:  plug virtual image (salmon) over socket virtual image (black)")):
        ax = fig.add_subplot(grid[row, :])
        ax.axis("off")
        ax.text(0.0, 0.15, text, fontsize=10, ha="left", va="bottom", transform=ax.transAxes)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
