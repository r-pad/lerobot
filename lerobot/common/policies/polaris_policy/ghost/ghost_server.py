"""
Ghost preprocessing server — runs in the `mapanything` conda env.

Pure depth-augmentation service: receives an RGB observation over websockets,
runs MapAnything to produce metric depth from the two camera views, and
returns the same obs with `depth1` / `depth2` (and undistorted `pixels1` /
`pixels2`) attached. The client is responsible for action prediction.

Setup (one-time):
    git clone https://github.com/facebookresearch/map-anything
    cd map-anything
    conda create -n mapanything python=3.12 -y
    conda activate mapanything
    pip install -e ".[all]"
    pip install msgpack msgpack-numpy websockets tyro

Run:
    conda activate mapanything
    python lerobot/common/policies/polaris_policy/ghost/ghost_server.py \\
        --port 8766 \\
        --calib /home/haotian/lerobot/lerobot/scripts/droid_calibration \\
        [--mapanything_model facebook/map-anything]

Calibration directory layout (4 txt files, see `load_calib`):
    intrinsics_front.txt           (3, 3)
    intrinsics_left.txt            (3, 3)
    T_world_from_camera_front.txt  (4, 4)  cam→world
    T_world_from_camera_left.txt   (4, 4)  cam→world

Wire protocol:
    request:  {"command": "preprocess",
               "obs":      {<front_key>: (H, W, 3) uint8 RGB — front cam,
                            <left_key>:  (H, W, 3) uint8 RGB — left  cam,
                            ...  (anything else is passed through unchanged)},
               "cam_keys": [<front_key>, <left_key>]   # required: original obs keys,
                                                       # ordering = [cam0=front, cam1=left]
                                                       # to match the calibration files.
               "return_viz": bool (optional, default False)}

    response: {"obs": {<front_key>:                   (1, 3, H, W) float32 [0, 1] — front RGB (normalized),
                       <left_key>:                    (1, 3, H, W) float32 [0, 1] — left  RGB (normalized),
                       <depth_key(front_key)>:        (H, W)    float32 metres — front depth (mm-quantized),
                       <depth_key(left_key)>:         (H, W)    float32 metres — left  depth (mm-quantized),
                       <intrinsics_key(front_key)>:   (3, 3)    float32 — front K,
                       <intrinsics_key(left_key)>:    (3, 3)    float32 — left  K,
                       <extrinsics_key(front_key)>:   (4, 4)    float32 — front world→cam,
                       <extrinsics_key(left_key)>:    (4, 4)    float32 — left  world→cam,
                       ...  (other keys from the request preserved verbatim)},
               "viz":  (H, 2W, 3) uint8 RGB | None (depth visualization if requested)}

    Derived key naming (for cam key e.g. `observation.images.cam_azure_kinect_front.color`):
      depth_key(k)      → `observation.images.cam_azure_kinect_front.transformed_depth`
      intrinsics_key(k) → `observation.cam_azure_kinect_front.intrinsics`
      extrinsics_key(k) → `observation.cam_azure_kinect_front.extrinsics`

    request:  {"command": "shutdown"}  →  {"status": "shutdown"}
"""

import asyncio
import dataclasses
import logging
import socket
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np
import torch
import tyro
import websockets


# CC BY-NC model (best indoor performance). Swap to "facebook/map-anything-apache"
# if Apache 2.0 licensing is required.
DEFAULT_MAPANYTHING_MODEL = "facebook/map-anything"


@dataclasses.dataclass
class Args:
    calib: str
    """Path to the calibration **directory** containing four txt files:
        intrinsics_front.txt           (3, 3)
        intrinsics_left.txt            (3, 3)
        T_world_from_camera_front.txt  (4, 4)  cam→world
        T_world_from_camera_left.txt   (4, 4)  cam→world
    """

    port: int = 8766
    """Websocket port to bind."""

    mapanything_model: str = DEFAULT_MAPANYTHING_MODEL
    """HF model id passed to MapAnything.from_pretrained."""

    device: str = "cuda"
    """Device for MapAnything inference."""


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def load_calib(calib_dir: str):
    """Load 4 txt-file calibration entries from a directory.

    Returns:
        K_front, K_left:                  (3, 3) float32  — intrinsics
        T_front, T_left:                  (4, 4) float32  — cam→world (C2W; what MapAnything wants)
        W2C_front, W2C_left:              (4, 4) float32  — world→cam (the "extrinsics" we expose)
    """
    d = Path(calib_dir)
    K_front = np.loadtxt(d / "intrinsics_front.txt").astype(np.float32)            # (3, 3)
    K_left  = np.loadtxt(d / "intrinsics_left.txt").astype(np.float32)             # (3, 3)
    T_front = np.loadtxt(d / "T_world_from_camera_front.txt").astype(np.float32)   # (4, 4) C2W
    T_left  = np.loadtxt(d / "T_world_from_camera_left.txt").astype(np.float32)    # (4, 4) C2W

    W2C_front = np.linalg.inv(T_front).astype(np.float32)
    W2C_left  = np.linalg.inv(T_left).astype(np.float32)
    return K_front, K_left, T_front, T_left, W2C_front, W2C_left


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def depth_to_viz(depth_m: np.ndarray, max_m: float = 2.0) -> np.ndarray:
    """Colorize (H, W) float32 depth (metres) → (H, W, 3) uint8 RGB for display."""
    d  = np.clip(depth_m, 0.0, max_m) / max(max_m, 1e-6)
    d8 = (d * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)[:, :, ::-1]  # BGR → RGB


def quantize_depth_mm(depth_m: np.ndarray) -> np.ndarray:
    """Round-trip float metres → uint16 mm → float metres.

    Matches lerobot's on-disk Azure-Kinect depth format (`* 1000 → uint16`,
    clipped to [0, 65535] mm = [0, 65.535] m). The forward + inverse cast keeps
    the dtype the consumer expects (float32 m) while applying the same
    quantization & range clipping the dataset would have applied.
    """
    return (np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16).astype(np.float32) / 1000.0)


# def normalize_rgb(img: np.ndarray) -> np.ndarray:
#     """(H, W, 3) uint8 → (1, 3, H, W) float32 [0, 1] (numpy, CPU).

#     Mirrors the user's `img_to_tensor` helper but stays in numpy so the result
#     is msgpack-serializable. The client can do `torch.from_numpy(arr).to(device)`
#     right before model invocation — no extra permute / unsqueeze needed.
#     """
#     arr = img.astype(np.float32) / 255.0          # (H, W, 3)
#     return np.transpose(arr, (2, 0, 1))[None]     # (1, 3, H, W)


def depth_key(color_key: str) -> str:
    """Derive the depth-feature key for a given color-feature key.

    `observation.images.X.color` → `observation.images.X.transformed_depth`
    Anything else just gets `.transformed_depth` appended.
    """
    suffix = ".color"
    if color_key.endswith(suffix):
        return color_key[: -len(suffix)] + ".transformed_depth"
    return color_key + ".transformed_depth"


def _cam_root_key(color_key: str) -> str:
    """`observation.images.X.color` → `observation.X` (drops `.images.` and `.color`).

    Used as a base for camera-level attribute keys (intrinsics, extrinsics).
    """
    suffix = ".color"
    base = color_key[: -len(suffix)] if color_key.endswith(suffix) else color_key
    return base.replace(".images.", ".")


def intrinsics_key(color_key: str) -> str:
    """Derive the camera intrinsics key.

    `observation.images.X.color` → `observation.X.intrinsics`
    """
    return _cam_root_key(color_key) + ".intrinsics"


def extrinsics_key(color_key: str) -> str:
    """Derive the world→cam extrinsics key.

    `observation.images.X.color` → `observation.X.extrinsics`
    """
    return _cam_root_key(color_key) + ".extrinsics"


# ──────────────────────────────────────────────────────────────────────────────
# MapAnything wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MapAnythingDepth:
    """Lazily-loaded MapAnything wrapper. Both views in one forward pass."""

    def __init__(self, model_id: str, device: str):
        from mapanything.models import MapAnything
        logging.info("Loading MapAnything model: %s (device=%s)", model_id, device)
        self.model  = MapAnything.from_pretrained(model_id).to(device).eval()
        self.device = device
        logging.info("MapAnything ready.")

    @torch.no_grad()
    def __call__(
        self,
        img0_rgb: np.ndarray,
        img1_rgb: np.ndarray,
        K0_new: np.ndarray,
        K1_new: np.ndarray,
        C2W0: np.ndarray,
        C2W1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        from mapanything.utils.image import preprocess_inputs

        views = [
            {
                "img":             img0_rgb,
                "intrinsics":      torch.tensor(K0_new, dtype=torch.float32),
                "camera_poses":    torch.tensor(C2W0,   dtype=torch.float32),
                "is_metric_scale": torch.tensor([True], device=self.device),
            },
            {
                "img":             img1_rgb,
                "intrinsics":      torch.tensor(K1_new, dtype=torch.float32),
                "camera_poses":    torch.tensor(C2W1,   dtype=torch.float32),
                "is_metric_scale": torch.tensor([True], device=self.device),
            },
        ]
        processed   = preprocess_inputs(views)
        predictions = self.model.infer(
            processed,
            memory_efficient_inference = True,
            use_amp                    = True,
            amp_dtype                  = "bf16",
            apply_mask                 = True,
            mask_edges                 = True,
            apply_confidence_mask      = False,
            ignore_calibration_inputs  = False,
            ignore_pose_inputs         = False,
            ignore_depth_inputs        = False,
        )
        depth0 = predictions[0]["depth_z"][0, :, :, 0].cpu().numpy().astype(np.float32)
        depth1 = predictions[1]["depth_z"][0, :, :, 0].cpu().numpy().astype(np.float32)

        # MapAnything outputs depth at its internal working resolution (e.g. 294x518
        # for a 16:9 input via RESOLUTION_MAPPINGS[518]). Upsample back to the input
        # RGB resolution so downstream consumers (compute_pcd, high_level wrapper)
        # see matching shapes.
        import torch.nn.functional as F
        h0, w0 = img0_rgb.shape[:2]
        h1, w1 = img1_rgb.shape[:2]
        if depth0.shape != (h0, w0):
            depth0 = F.interpolate(
                torch.from_numpy(depth0)[None, None], size=(h0, w0),
                mode="nearest",
            )[0, 0].numpy().astype(np.float32)
        if depth1.shape != (h1, w1):
            depth1 = F.interpolate(
                torch.from_numpy(depth1)[None, None], size=(h1, w1),
                mode="nearest",
            )[0, 0].numpy().astype(np.float32)
        return depth0, depth1


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing server
# ──────────────────────────────────────────────────────────────────────────────

class GhostServer:
    def __init__(self, args: Args):
        calib_dir = Path(args.calib)
        if not calib_dir.is_dir():
            raise NotADirectoryError(f"calib must be a directory of txt files: {calib_dir}")
        K_front, K_left, T_front, T_left, W2C_front, W2C_left = load_calib(str(calib_dir))
        # cam0 = front, cam1 = left (matches calibration directory file naming).
        self.K0,   self.K1   = K_front,   K_left
        self.C2W0, self.C2W1 = T_front,   T_left       # cam→world (for MapAnything)
        self.W2C0, self.W2C1 = W2C_front, W2C_left     # world→cam (exposed as "extrinsics")
        logging.info("Calibration loaded: cam0 fx=%.1f, cam1 fx=%.1f", K_front[0, 0], K_left[0, 0])

        self.mapany = MapAnythingDepth(args.mapanything_model, args.device)
        logging.info("GhostServer ready.")

    def preprocess(
        self,
        obs: dict,
        cam_keys: list[str],
        return_viz: bool = False,
    ) -> tuple[dict, np.ndarray | None]:
        """Undistort the RGB pair and add MapAnything depth to `obs`, preserving
        the caller's original camera keys.

        Args:
            obs:      observation dict from the client. Must contain the two camera
                      keys named in `cam_keys`. All other keys are passed through.
            cam_keys: [front_key, left_key] — ordering matches calibration (cam0=front,
                      cam1=left).
            return_viz: if True, also return a side-by-side depth heatmap.

        Mutates `obs` in place:
          obs[front_key] / obs[left_key]   : (1, 3, H, W) float32 [0, 1] — normalized RGB
          obs[depth_key(front_key)]        : (H, W) float32 metres        — front depth
                                             (mm-quantized: m → uint16 mm → m, lerobot Azure-Kinect format)
          obs[depth_key(left_key)]         : (H, W) float32 metres        — left  depth (same quantization)
          obs[intrinsics_key(front_key)]   : (3, 3) float32               — front K
          obs[intrinsics_key(left_key)]    : (3, 3) float32               — left  K
          obs[extrinsics_key(front_key)]   : (4, 4) float32               — front world→cam
          obs[extrinsics_key(left_key)]    : (4, 4) float32               — left  world→cam
        """
        if not isinstance(cam_keys, (list, tuple)) or len(cam_keys) != 2:
            raise ValueError(
                f"cam_keys must be a 2-element list [front_key, left_key]; got {cam_keys!r}"
            )
        front_key, left_key = cam_keys[0], cam_keys[1]
        for k in (front_key, left_key):
            if k not in obs:
                raise KeyError(
                    f"cam_keys references '{k}' but it's not in obs (keys: {list(obs.keys())})"
                )

        img0 = np.asarray(obs[front_key], dtype=np.uint8)
        img1 = np.asarray(obs[left_key],  dtype=np.uint8)

        # MapAnything: pass intrinsics and C2W directly (no undistortion).
        depth0_m, depth1_m = self.mapany(
            img0, img1, self.K0, self.K1, self.C2W0, self.C2W1
        )

        # Round-trip depth through uint16 mm so the values match the precision /
        # range the lerobot dataset would have stored (Azure Kinect convention).
        depth0_m = quantize_depth_mm(depth0_m)
        depth1_m = quantize_depth_mm(depth1_m)


        # Attach depth, intrinsics (K), and world→cam extrinsics under derived
        # names so the client can locate them by the camera's color key.
        new_obs = {}
        new_obs["front"]      = depth0_m
        new_obs["left"]       = depth1_m


        viz = None
        if return_viz:
            v0 = depth_to_viz(depth0_m)
            v1 = depth_to_viz(depth1_m)
            h  = min(v0.shape[0], v1.shape[0])
            if v0.shape[0] != h:
                v0 = cv2.resize(v0, (v0.shape[1], h))
            if v1.shape[0] != h:
                v1 = cv2.resize(v1, (v1.shape[1], h))
            viz = np.concatenate([v0, v1], axis=1)

        return new_obs, viz


# ──────────────────────────────────────────────────────────────────────────────
# Websocket loop
# ──────────────────────────────────────────────────────────────────────────────

def main(args: Args):
    server         = GhostServer(args)
    shutdown_event = asyncio.Event()

    async def handle(websocket):
        async for message in websocket:
            data    = msgpack.unpackb(message, raw=False)
            command = data.get("command", "preprocess")

            if command == "preprocess":
                obs        = data["obs"]
                cam_keys   = data.get("cam_keys")
                return_viz = bool(data.get("return_viz", False))
                if cam_keys is None:
                    await websocket.send(msgpack.packb(
                        {"error": "request missing required field 'cam_keys' (list of 2 obs keys)"},
                        use_bin_type=True,
                    ))
                    continue
                try:
                    obs_out, viz = server.preprocess(obs, cam_keys, return_viz=return_viz)
                    await websocket.send(
                        msgpack.packb({"obs": obs_out, "viz": viz}, use_bin_type=True)
                    )
                except Exception as e:
                    import traceback
                    logging.error("preprocess failed: %s\n%s", e, traceback.format_exc())
                    await websocket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))

            elif command == "shutdown":
                logging.info("Port %d: shutdown signal received.", args.port)
                await websocket.send(msgpack.packb({"status": "shutdown"}, use_bin_type=True))
                shutdown_event.set()
                return

            else:
                await websocket.send(
                    msgpack.packb({"error": f"Unknown command: {command}"}, use_bin_type=True)
                )

    async def serve():
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Serving on host=%s ip=%s port=%d", hostname, local_ip, args.port)

        async with websockets.serve(handle, "0.0.0.0", args.port, max_size=100 * 1024 * 1024):
            await shutdown_event.wait()

        logging.info("Port %d: server shut down.", args.port)

    asyncio.run(serve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
