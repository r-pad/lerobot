"""
GhostClient: websockets/msgpack client that calls ghost_server.py for depth
preprocessing (RGB → depth via MapAnything) and then runs the actual ghost
action-prediction model **locally** in the lerobot env.

The ghost server only does MapAnything depth augmentation — it does not load
or run any policy weights. Action prediction happens here in the client,
loaded from `--control.policy.path` (the standard lerobot pretrained path,
exposed as `pretrained_path` on the config).

Integration with control_robot.py:
    python lerobot/scripts/control_robot.py \\
        --robot.type=droid \\
        --control.type=record \\
        --control.policy.type=ghost \\
        --control.policy.path=/path/to/ghost_local_model_dir \\
        --control.policy.host=localhost \\
        --control.policy.port=8766 \\
        --control.policy.cam_image_keys='["observation.images.cam_azure_kinect_front.color",
                                          "observation.images.cam_azure_kinect_left.color"]' \\
        --control.policy.image_resize='[720, 1280]' \\
        --control.policy.eef_pose_key=observation.right_eef_pose \\
        --control.policy.transform_eef_to_states_ee=true \\
        --control.policy.use_ik=true \\
        ...

Server wire protocol (preprocessing only):
    request:  {"command": "preprocess",
               "obs":  per-cam RGB images (under cam_image_keys), states_ee, gripper_pcd, etc.,
               "cam_keys": [<front_cam_key>, <left_cam_key>] — the two views to depth-augment,
               "return_viz": bool}
    response: {"obs":  the request obs round-tripped, plus one `*_depth` key per cam_key
                       (e.g. "observation.images.cam_azure_kinect_front.transformed_depth"),
               "viz":  (H, 2W, 3) uint8 RGB | None}

    request:  {"command": "shutdown"}  →  {"status": "shutdown"}
"""

import asyncio
import os
from dataclasses import dataclass, field

import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np
import torch
import torch.nn.functional as F
import websockets
from torch import Tensor

import lerobot
from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("ghost")
@dataclass
class GhostConfig(PreTrainedConfig):
    """Configuration for GhostClient (websockets client to ghost_server.py).

    Args:
        host: Hostname of the machine running ghost_server.py.
        port: websockets port that ghost_server.py is listening on.
        cam_image_keys: Ordered list of two lerobot camera observation keys to use as
            ghost views [pixels1, pixels2] — pass them in [front, left] order to
            match the calibration JSON (cam0=front, cam1=left).
        eef_pose_key: lerobot batch key holding the eef pose
            (shape (10,) in lerobot format [rot6d(6)|trans(3)|gripper(1)]).
        image_resize: (H, W) to resize camera images before sending. If None, sent at native size.
        transform_eef_to_states_ee: If True, convert lerobot eef_pose
            [rot6d(6)|trans(3)|gripper(1)] to polaris states_ee
            [trans(3)|quat_wxyz(4)|gripper(1)].
        use_ik: If True, convert EEF action → joint positions via DroidAdapter IK.
        undo_z_rotation_deg: Undo training-time Z-rotation augmentation on actions
            and apply the inverse on observations.
        vis_tracks: If True, request visualization frames from the server.
        local_policy_path: Directory of a trained DiffusionPolicy checkpoint
            (config.json + model.safetensors). Used instead of --control.policy.path
            because the lerobot parser's path-handling conflicts with
            --control.policy.type=ghost. When set, the diffusion-specific fields are
            synced from the checkpoint's config.json so the architecture matches, and
            the safetensors weights are loaded into the underlying DiffusionPolicy.
    """

    host: str = "localhost"
    port: int = 8766
    cam_image_keys: list = field(default_factory=list)
    vis_obs_key: str = "observation.images.ghost_vis"
    # (H, W*num_views, 3) shape of the per-step ghost viz frame. The viz is the front
    # and left Azure-Kinect RGBs (each at the original 720x1280) side-by-side, each
    # blended with the high-level goal-gripper heatmap → defaults to 720x2560. Override
    # on the CLI if image_resize changes the per-cam dims.
    vis_shape: list = field(default_factory=lambda: [720, 2560, 3])
    eef_pose_key: str = "observation.right_eef_pose"
    image_resize: list | None = None
    # Server expects polaris-format states_ee [trans(3)|quat_wxyz(4)|gripper(1)] (indexed
    # at [3:7] for quat). lerobot format is 10-d rot6d+trans+gripper, so this conversion
    # is required for the server to read the obs correctly.
    transform_eef_to_states_ee: bool = True
    use_ik: bool = True
    # Dataset was preprocessed with -45° z-axis world-frame rotation on the rotation part
    # (commit bb6730e). Default 45.0 undoes that on the action side and applies the inverse
    # (-45°) on the obs side so the policy sees rotations in the same frame as training.
    undo_z_rotation_deg: float = 45.0
    vis_tracks: bool = True
    enable_goal_conditioning: bool = False  # compatibility with control_robot.py
    local_policy_path: str | None = None
    use_map_anything: bool = True

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert an (H, W, C) uint8 image to a (1, C, H, W) float32 tensor in [0, 1].

    Matches the lerobot visual-feature convention so downstream policies (which
    declare VISUAL inputs of shape [C, H, W] and apply MEAN_STD / MIN_MAX
    normalization on float values in [0, 1]) can consume it directly.
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"img_to_tensor: expected (H, W, C), got shape {img.shape}")
    t = torch.from_numpy(img).to(torch.float32) / 255.0   # (H, W, C) float32 [0, 1]
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()    # (1, C, H, W)


def mix_depth_gt_and_predicted(
    gt_depth_m: np.ndarray,
    predicted_depth_m: np.ndarray,
    *,
    gt_invalid_threshold_m: float = 0.0,
) -> np.ndarray:
    """Fill holes in the ground-truth depth using a predicted depth map.

    Where GT is "valid" (> threshold) keep GT. Where GT is missing/zero, fall back
    to the predicted depth. Useful for combining a Kinect's `transformed_depth`
    (accurate but full of holes, especially on shiny / dark / out-of-range surfaces)
    with a MapAnything prediction (smooth, dense, less accurate at edges).

    Args:
        gt_depth_m:        (H, W) float32 depth in metres. 0 / NaN / negatives count as invalid.
        predicted_depth_m: (H, W) float32 depth in metres. Same shape as `gt_depth_m`.
        gt_invalid_threshold_m: GT depths ≤ this value are replaced by predicted.

    Returns:
        (H, W) float32 — GT where valid, predicted where GT is invalid.
    """
    gt = np.asarray(gt_depth_m, dtype=np.float32)
    pr = np.asarray(predicted_depth_m, dtype=np.float32)
    if gt.shape != pr.shape:
        raise ValueError(
            f"mix_depth_gt_and_predicted: shape mismatch gt={gt.shape} pr={pr.shape}"
        )
    valid = np.isfinite(gt) & (gt > gt_invalid_threshold_m)
    return np.where(valid, gt, pr).astype(np.float32)


def backproject_rgbd_to_world(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    *,
    stride: int = 4,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an RGB-D frame to a world-frame coloured point cloud.

    Args:
        rgb:         (H, W, 3) uint8 image.
        depth_m:     (H, W) float depth in metres (0 / NaN = invalid).
        K:           (3, 3) camera intrinsics for this RGB resolution.
        cam_to_world:(4, 4) extrinsics that take camera-frame points to world.
        stride:      pixel stride for subsampling (4 → 1/16 of pixels).
        min/max_depth_m: depth clipping range to drop sky / hand-rays.

    Returns:
        pts_world: (N, 3) float32, world-frame xyz.
        cols:      (N, 3) uint8, matching RGB.
    """
    h, w = depth_m.shape[:2]
    # Subsample on a regular pixel grid for log speed.
    vs, us = np.mgrid[0:h:stride, 0:w:stride].astype(np.float32)
    d = depth_m[::stride, ::stride].astype(np.float32)
    mask = np.isfinite(d) & (d > min_depth_m) & (d < max_depth_m)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    us, vs, d = us[mask], vs[mask], d[mask]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_c = (us - cx) * d / fx
    y_c = (vs - cy) * d / fy
    z_c = d
    pts_cam = np.stack([x_c, y_c, z_c], axis=1)  # (N, 3)

    # cam → world: world = R * cam + t.
    R = cam_to_world[:3, :3].astype(np.float32)
    t = cam_to_world[:3, 3].astype(np.float32)
    pts_world = pts_cam @ R.T + t  # (N, 3)

    cols = rgb[::stride, ::stride][mask]  # (N, 3) uint8
    return pts_world.astype(np.float32), cols.astype(np.uint8)


def get_heatmap_viz(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a 1-channel distance heatmap on an RGB image (goal center pops red).

    Args:
        rgb_image: (H, W, 3) uint8 RGB.
        heatmap:   (H, W) uint8 — distance to the goal point (higher = farther).
        alpha:     heatmap blend weight in [0, 1].

    Returns:
        (H, W, 3) uint8 RGB blend.
    """
    import cv2
    gray = np.asarray(heatmap, dtype=np.uint8)
    gray_inv = 255 - gray  # invert so the goal point (distance 0) becomes the brightest
    colored_bgr = cv2.applyColorMap(gray_inv, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blended = rgb_image.astype(np.float32) * (1 - alpha) + colored_rgb.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def draw_gripper_pcd_overlay(
    rgb_image: np.ndarray,
    pcd_world: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Project a world-frame gripper PCD into the image plane and label the 4 points.

    Points are labelled in the order produced by _extract_obs after the [[1, 2, 0, 3]]
    reorder — i.e. [right, left, top, grasp]. K is auto-scaled to the actual
    rgb_image (H, W) using its principal point (cx ≈ W/2, cy ≈ H/2 implies the K
    was calibrated at (2*cy, 2*cx)).

    Args:
        rgb_image: (H, W, 3) uint8.
        pcd_world: (4, 3) float — world-frame gripper points.
        K:         (3, 3) intrinsics.
        cam_to_world: (4, 4) extrinsic (T_world_from_camera).
        font_scale: cv2.putText scale factor.

    Returns:
        (H, W, 3) uint8 — `rgb_image` with the 4 points labelled by name.
    """
    import cv2
    img = rgb_image.copy()
    H, W = img.shape[:2]

    K = np.asarray(K, dtype=np.float64).copy()
    H_calib = K[1, 2] * 2.0
    W_calib = K[0, 2] * 2.0
    if abs(H_calib - H) > 1 or abs(W_calib - W) > 1:
        sx = W / W_calib
        sy = H / H_calib
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, dtype=np.float64))
    pts_cam = (world_to_cam[:3, :3] @ pcd_world.T).T + world_to_cam[:3, 3]
    valid = pts_cam[:, 2] > 1e-6
    z_safe = np.where(valid[:, None], pts_cam[:, 2:3], 1.0)
    pts_2d = (K @ pts_cam.T).T[:, :2] / z_safe

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (pt, ok) in enumerate(zip(pts_2d, valid)):
        if not ok:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < W and 0 <= y < H):
            continue
        # White filled dot with a black outline so it stays visible on any backdrop.
        cv2.circle(img, (x, y), 6, (0, 0, 0),       thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        # Small index label (0=right, 1=left, 2=top, 3=grasp) beside the dot.
        label = str(i)
        text_org = (x + 8, y - 6)
        cv2.putText(img, label, text_org, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, text_org, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load_policy(policy_path: str):

    policy_config: PreTrainedConfig = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path

    lerobot_dir = os.path.dirname(os.path.dirname(lerobot.__file__))
    if hasattr(policy_config, "calibration_json") and policy_config.calibration_json:
        policy_config.calibration_json = os.path.join(lerobot_dir, policy_config.calibration_json)

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(config=policy_config, pretrained_name_or_path=policy_path)
    policy.eval()
    policy.to("cuda")
    return policy

class GhostClient(PreTrainedPolicy):
    """Thin websockets client that forwards observations to ghost_server.py."""

    config_class = GhostConfig
    name = "ghost"

    def __init__(self, config: GhostConfig, **kwargs):
        super().__init__(config)
        self.config = config

        self.uri = f"ws://{config.host}:{config.port}"
        print(f"[GhostClient] Server URI (preprocessing only): {self.uri}")
        print(f"[GhostClient] Camera views: {config.cam_image_keys}")

        self.policy = load_policy(config.local_policy_path)

        if config.use_ik:
            from lerobot.common.policies.robot_adapters import DroidAdapter
            self._droid_adapter = DroidAdapter(action_space="right_eef")
        else:
            self._droid_adapter = None

        self._current_vis_frame: np.ndarray | None = None  # (H, W*v, 3) uint8 RGB

        # Cached high-level goal-gripper heatmaps from the most recent
        # predict_and_project call — re-used across the action chunk while
        # enable_goal_condition gates re-prediction.
        self._latest_goal_heatmap_front: np.ndarray | None = None  # (H, W, 3) uint8
        self._latest_goal_heatmap_left:  np.ndarray | None = None  # (H, W, 3) uint8

        # Caches for action-chunk replay: the ghost server depth call and the
        # high-level goal prediction are expensive, so we only re-run them on
        # steps where the inner DiffusionPolicy's action queue is empty (i.e.
        # the start of a fresh chunk). Subsequent steps reuse these.
        self._cached_depth_left_cpu:   torch.Tensor | None = None  # (1, H, W) float metres
        self._cached_depth_front_cpu:  torch.Tensor | None = None  # (1, H, W) float metres
        self._cached_depth_left_u16:   torch.Tensor | None = None  # (1, H, W) uint16 mm
        self._cached_depth_front_u16:  torch.Tensor | None = None  # (1, H, W) uint16 mm

    async def _send(self, message: dict) -> dict:
        async with websockets.connect(self.uri, max_size=100 * 1024 * 1024) as ws:
            await ws.send(msgpack.packb(message, use_bin_type=True))
            response = await ws.recv()
            return msgpack.unpackb(response, raw=False)

    def reset(self):
        # The underlying DiffusionPolicy keeps action/obs queues in self._queues;
        # forward to it so they're (re)initialized. The ghost server itself is
        # stateless (pure preprocessing) — no reset command needed.
        self.policy.reset()
        self._current_vis_frame = None
        self._latest_goal_heatmap_front = None
        self._latest_goal_heatmap_left = None
        self._cached_depth_left_cpu = None
        self._cached_depth_front_cpu = None
        self._cached_depth_left_u16 = None
        self._cached_depth_front_u16 = None

    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Build obs → send to server for depth preprocessing → run local
        action-prediction model on the depth-augmented obs → return (action, action_eef).

        Returns:
            action: (1, action_dim) joint positions or EEF depending on use_ik.
            action_eef: (1, 10) lerobot EEF format [rot6d(6)|trans(3)|gripper_lerobot(1)].
        """
        obs = self._extract_obs(batch)

        batch_obs = batch.copy()
        device = self.policy.config.device

        left_depth_key = "observation.images.cam_azure_kinect_left.transformed_depth"
        front_depth_key = "observation.images.cam_azure_kinect_front.transformed_depth"

        # === MapAnything-depth gate ===
        # When self.config.use_map_anything is False, skip the server call and
        # the depth-replacement entirely. The camera-native transformed_depth
        # already sits on the outer observation dict and will flow through to
        # dataset.add_frame on its own (we just leave _extra_observation empty).
        if not self.config.use_map_anything:
            depth_left_cpu  = None
            depth_front_cpu = None
            self._extra_observation = {}
        else:
            # === Caching gate ===
            # The ghost-server depth call (MapAnything roundtrip) and the high-level
            # goal-prediction call are by far the two most expensive operations per
            # step. The inner DiffusionPolicy generates a chunk of `n_action_steps`
            # actions on the step where its action queue is empty, and dequeues one
            # per call thereafter — so on those intermediate "replay" steps the
            # action is already chosen and we can reuse last chunk's depth +
            # heatmaps without changing the policy's behaviour.
            queue_empty = (
                self.policy._queues is None
                or len(self.policy._queues[self.policy.act_key]) == 0
            )
            # On the very first call there's no cached depth yet, so force a fresh
            # plan regardless of queue state.
            no_cached_depth = self._cached_depth_left_u16 is None
            run_fresh_plan = queue_empty or no_cached_depth

            if run_fresh_plan:
                result = asyncio.run(self._send({
                    "command": "preprocess",
                    "obs": obs,
                    # The server only does depth estimation for the two kinect views; it
                    # expects exactly 2 cam_keys ordered [cam0=front, cam1=left] (matching
                    # the calibration JSON). Any additional cam_image_keys (e.g. cam_wrist)
                    # are policy-only inputs and are not sent to the server for depth.
                    "cam_keys": list(self.config.cam_image_keys)[:2],
                    # We build the ghost viz client-side (heatmap overlay) below, so don't
                    # pay the cost of having the server render anything.
                    "return_viz": False,
                }))
                if "error" in result:
                    raise RuntimeError(f"[GhostClient] Server error: {result['error']}")

                mapanything_obs = result["obs"]
                pred_left_2d  = np.asarray(mapanything_obs["left"],  dtype=np.float32)  # (H, W) metres
                pred_front_2d = np.asarray(mapanything_obs["front"], dtype=np.float32)

                # Hole-fill the kinect's native transformed_depth (camera depth has
                # zeros on shiny / dark / out-of-range surfaces) with MapAnything's
                # smooth-but-less-accurate prediction. Camera depth survives at every
                # pixel where it read a finite, > 0 value.
                def _squeeze_to_hw(t: torch.Tensor) -> np.ndarray:
                    arr = t.detach().cpu().numpy().astype(np.float32)
                    while arr.ndim > 2 and arr.shape[0] == 1:
                        arr = arr.squeeze(0)
                    return arr

                if left_depth_key in batch_obs:
                    gt_left_2d  = _squeeze_to_hw(batch_obs[left_depth_key])
                    mixed_left_2d  = mix_depth_gt_and_predicted(gt_left_2d,  pred_left_2d)
                else:
                    mixed_left_2d  = pred_left_2d
                if front_depth_key in batch_obs:
                    gt_front_2d = _squeeze_to_hw(batch_obs[front_depth_key])
                    mixed_front_2d = mix_depth_gt_and_predicted(gt_front_2d, pred_front_2d)
                else:
                    mixed_front_2d = pred_front_2d

                depth_left_cpu  = torch.from_numpy(pred_left_2d).unsqueeze(0)   # (1, H, W) float metres
                depth_front_cpu = torch.from_numpy(pred_front_2d).unsqueeze(0)
                # uint16 mm for the dataset writer (the only dtype its image_array_to_pil
                # path supports for 2-D depth images).
                depth_left_u16  = (depth_left_cpu.float() * 1000.0).clamp(0, 65535).to(torch.uint16)
                depth_front_u16 = (depth_front_cpu.float() * 1000.0).clamp(0, 65535).to(torch.uint16)

                self._cached_depth_left_cpu   = depth_left_cpu
                self._cached_depth_front_cpu  = depth_front_cpu
                self._cached_depth_left_u16   = depth_left_u16
                self._cached_depth_front_u16  = depth_front_u16
            else:
                depth_left_cpu   = self._cached_depth_left_cpu
                depth_front_cpu  = self._cached_depth_front_cpu
                depth_left_u16   = self._cached_depth_left_u16
                depth_front_u16  = self._cached_depth_front_u16

            # Both kinect transformed_depth keys are declared in robot.features (the
            # cameras have use_transformed_depth=true), so the dataset schema accepts
            # both. _extra_observation overwrites the camera-native depth with the
            # MapAnything depth in the recorded frame; cached steps re-emit the same
            # uint16 tensors so the dataset has a depth video frame every tick.
            if left_depth_key in batch_obs.keys():
                del batch_obs[left_depth_key]
            if front_depth_key in batch_obs.keys():
                del batch_obs[front_depth_key]
            batch_obs[left_depth_key]  = depth_left_cpu.to(device)
            batch_obs[front_depth_key] = depth_front_cpu.to(device)
            self._extra_observation = {
                left_depth_key:  depth_left_u16,
                front_depth_key: depth_front_u16,
            }

        # Overlay the 10-d polaris-layout eef_pose [pos(3) | rot6d(6) | gripper_polaris(1)]
        # built by _extract_obs (rotated training frame, polaris gripper convention).
        # obs holds it as float32 numpy (wire-serializable); re-tensorize here.
        batch_obs["observation.right_eef_pose"] = (
            torch.as_tensor(obs["observation.right_eef_pose"]).to(device)
        )

        # predict_and_project expects RGB as (H, W, 3) uint8 and depth as a 2D float array
        # in millimetres (matches the on-disk Azure-Kinect transformed_depth convention used
        # at training). batch_obs tensors are (1, 3, H, W) float [0,1] / (1, H, W) float m.
        def _rgb_to_np(t: torch.Tensor) -> np.ndarray:
            arr = t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (arr * 255.0).clip(0, 255).astype(np.uint8)

        def _depth_to_np(t: torch.Tensor) -> np.ndarray:
            # MapAnything path: t is (1, H, W) float metres.
            # Camera path (use_map_anything=False): image-prep wrapped the
            # camera's (H, W, 1) uint16 mm → (1, 1, H, W) float metres.
            # Squeeze leading size-1 dims until we get the bare (H, W) the
            # high-level wrapper expects, then scale m → mm.
            arr = t.detach().cpu().numpy().astype(np.float32)
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            return arr * 1000.0

        # Extract the front/left RGB frames once — used for both predict_and_project
        # and the ghost-viz blend below.
        front_rgb_np = _rgb_to_np(batch_obs["observation.images.cam_azure_kinect_front.color"])
        left_rgb_np  = _rgb_to_np(batch_obs["observation.images.cam_azure_kinect_left.color"])

        with torch.inference_mode():
            cam_name_front = self.policy.high_level.camera_names[0]
            cam_name_left  = self.policy.high_level.camera_names[1]

            enable_goal_condition = (
                hasattr(self.policy.config, "enable_goal_conditioning")
                and self.policy.config.enable_goal_conditioning
                and (self.policy._queues is None or len(self.policy._queues[self.policy.act_key]) == 0)
            )
            if enable_goal_condition:
                camera_obs = {
                    cam_name_front: {
                        "rgb":   front_rgb_np,
                        "depth": _depth_to_np(batch_obs["observation.images.cam_azure_kinect_front.transformed_depth"]),
                    },
                    cam_name_left: {
                        "rgb":   left_rgb_np,
                        "depth": _depth_to_np(batch_obs["observation.images.cam_azure_kinect_left.transformed_depth"]),
                    },
                }
                robot_kwargs = {
                    "observation.state": batch_obs["observation.state"],
                    "observation.right_eef_pose": batch_obs[self.config.eef_pose_key],
                    "gripper_pcd": obs["observation.points.gripper_pcds"],  # raw (4, 3) numpy
                }
                # control_utils.predict_action wraps str obs values in a 1-element list
                # for batching (control_utils.py:341); _get_text_embedding uses the text
                # as a dict key and needs the plain string.
                task = batch_obs["task"]
                if isinstance(task, list):
                    task = task[0]

                goal_pcds = self.policy.high_level.predict(task, camera_obs, robot_type=self.policy.config.robot_type, robot_kwargs=robot_kwargs)
                gripper_projs = self.policy.high_level.predict_and_project(
                    task,
                    camera_obs,
                    robot_type=self.policy.config.robot_type,
                    robot_kwargs=robot_kwargs,
                )
                for cam_name, proj in gripper_projs.items():
                    self.policy.latest_gripper_proj[cam_name] = img_to_tensor(proj).to(device)
                # Cache the raw (H, W, 3) uint8 heatmaps for the ghost-viz overlay.
                # high_level.camera_names is the source of truth for the front/left keys.
                self._latest_goal_heatmap_front = gripper_projs.get(cam_name_front)
                self._latest_goal_heatmap_left  = gripper_projs.get(cam_name_left)
                batch_obs["observation.points.goal_gripper_pcds"] = (
                    torch.as_tensor(goal_pcds).unsqueeze(0).to(device)
                )

            # The CURRENT 4-point gripper pcd (synthesized by _extract_obs from
            # the live eef_pose) is needed every step — both by the policy
            # (input_features declares `observation.points.gripper_pcds` as a
            # STATE feature) and by the viz overlay below. Add the batch dim so
            # populate_queues can stack along dim 1.
            batch_obs["observation.points.gripper_pcds"] = (
                torch.as_tensor(obs["observation.points.gripper_pcds"]).unsqueeze(0).to(device)
            )

            for cam_name, proj_tensor in self.policy.latest_gripper_proj.items():
                batch_obs[f"observation.images.{cam_name}.goal_gripper_proj"] = proj_tensor.to(device)

            action_joint, action_eef = self.policy.select_action(batch_obs)

        # Build the ghost viz: front/left RGBs with the cached goal heatmaps overlaid
        # (channel 0 = distance to the gripper-tip point), the current world-frame
        # gripper_pcd stamped as colored dots, then concatenated side-by-side. This is
        # written every step so the dataset records a frame-aligned viz even when the
        # high-level heatmap is reused mid-chunk.
        front_viz = front_rgb_np
        left_viz  = left_rgb_np
        if self._latest_goal_heatmap_front is not None:
            front_viz = get_heatmap_viz(front_rgb_np, self._latest_goal_heatmap_front[:, :, 0])
        if self._latest_goal_heatmap_left is not None:
            left_viz = get_heatmap_viz(left_rgb_np, self._latest_goal_heatmap_left[:, :, 0])

        # Overlay the gripper_pcd (right=red, left=green, top=blue, grasp=yellow). The
        # PCD is still in the rotated training frame on the policy side, so it lines up
        # with what the model consumes — not the raw world frame. high_level stores
        # original_Ks / cam_to_worlds parallel to camera_names ([front, left]).
        gripper_pcd_world = batch_obs["observation.points.gripper_pcds"].squeeze(0).cpu().numpy()
        front_viz = draw_gripper_pcd_overlay(
            front_viz, gripper_pcd_world,
            self.policy.high_level.original_Ks[0],
            self.policy.high_level.cam_to_worlds[0],
        )
        left_viz = draw_gripper_pcd_overlay(
            left_viz, gripper_pcd_world,
            self.policy.high_level.original_Ks[1],
            self.policy.high_level.cam_to_worlds[1],
        )

        self._current_vis_frame = np.concatenate([front_viz, left_viz], axis=1)

        # ── Rerun 3D viz ───────────────────────────────────────────────────────
        # Scene pcd from the high-level wrapper (compute_pcd ran on the last fresh
        # plan step), current gripper pcd (4 colored points), and the model-predicted
        # goal gripper pcd from the most recent HL forward pass. Logged every step so
        # the viewer always has a frame to render; the scene + goal points are stale
        # across cached steps which is intentional — they reflect what the policy
        # planned the chunk against.
        try:
            import rerun as rr
            # Scene point cloud — back-project the live RGB + depth using the
            # per-camera intrinsics and cam→world extrinsics from high_level. We
            # rebuild it every step (rather than reusing high_level.last_pcd_xyz)
            # so the viz reflects the camera's current view, even during chunk replay.
            #
            # Depth source depends on the flag: MapAnything (cached, shape (1, H, W))
            # when use_map_anything=True, else the camera-native transformed_depth
            # still sitting in batch_obs (shape (1, 1, H, W) float metres after the
            # predict_action image-prep). Reduce both to (H, W) via _depth_to_np-style
            # leading-1 squeeze so backproject_rgbd_to_world gets a clean 2-D map.
            def _scene_depth_m(cached_cpu, batch_key):
                if cached_cpu is not None:
                    arr = cached_cpu.cpu().numpy().astype(np.float32)
                elif batch_key in batch_obs:
                    arr = batch_obs[batch_key].detach().cpu().numpy().astype(np.float32)
                else:
                    return None
                while arr.ndim > 2 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                return arr  # (H, W) float metres

            depth_front_m = _scene_depth_m(depth_front_cpu, front_depth_key)
            depth_left_m  = _scene_depth_m(depth_left_cpu,  left_depth_key)
            if depth_front_m is not None and depth_left_m is not None:
                cam_to_worlds  = self.policy.high_level.cam_to_worlds
                intrinsics_all = self.policy.high_level.original_Ks
                front_pts, front_cols = backproject_rgbd_to_world(
                    front_rgb_np, depth_front_m, intrinsics_all[0], cam_to_worlds[0],
                )
                left_pts,  left_cols  = backproject_rgbd_to_world(
                    left_rgb_np,  depth_left_m,  intrinsics_all[1], cam_to_worlds[1],
                )
                # Log each camera's back-projection under its own entity path so they
                # can be toggled independently in the viewer.
                if front_pts.shape[0]:
                    rr.log(
                        "world/scene_pointcloud/cam_azure_kinect_front",
                        rr.Points3D(front_pts, colors=front_cols, radii=0.003),
                    )
                if left_pts.shape[0]:
                    rr.log(
                        "world/scene_pointcloud/cam_azure_kinect_left",
                        rr.Points3D(left_pts, colors=left_cols, radii=0.003),
                    )

            # 4-point gripper PCDs: [right, left, top, grasp] after our axis-0
            # permutation. Solid colors per pcd to distinguish current vs goal:
            # current = white, goal = red.
            _CUR_COLOR  = np.array([255, 255, 255], dtype=np.uint8)
            _GOAL_COLOR = np.array([255,   0,   0], dtype=np.uint8)

            # Current gripper pcd — re-read from batch_obs (still (1, 4, 3) here).
            gp_cur = batch_obs["observation.points.gripper_pcds"].squeeze(0).detach().cpu().numpy()
            rr.log(
                "world/current_gripper_pcd",
                rr.Points3D(gp_cur, colors=np.tile(_CUR_COLOR, (gp_cur.shape[0], 1)), radii=0.012),
            )

            # Goal gripper pcd from the HL prediction (if available).
            gp_goal_t = batch_obs.get("observation.points.goal_gripper_pcds", None)
            if gp_goal_t is not None:
                gp_goal = gp_goal_t.squeeze(0).detach().cpu().numpy()
                # last 4 points convention if HL emits more — use the points axis verbatim
                # otherwise (shape (4, 3) on this path).
                if gp_goal.shape[0] >= 4:
                    gp_goal_4 = gp_goal[-4:]
                    rr.log(
                        "world/goal_gripper_pcd",
                        rr.Points3D(
                            gp_goal_4,
                            colors=np.tile(_GOAL_COLOR, (gp_goal_4.shape[0], 1)),
                            radii=0.012,
                        ),
                    )
        except Exception as e:
            print(f"[ghost rerun] viz log failed: {e}")

        # action_eef is (1, 10) polaris layout [pos(3) | rot6d(6) | gripper_polaris(1)],
        # matching the input eef_pose layout the policy was trained on.
        action_ee_np = action_eef.squeeze(0).cpu().numpy()
        if action_ee_np.shape[0] != 10:
            raise ValueError(f"Expected action_eef shape (10,), got {action_ee_np.shape}")
        pos             = torch.from_numpy(action_ee_np[0:3]) # + torch.from_numpy(np.array([0.01, 0.0, 0.01], dtype=np.float32))
        rot6d    = torch.from_numpy(action_ee_np[3:9])
        gripper_polaris = torch.from_numpy(action_ee_np[9:10])  # 0=open, 1=closed

        # Undo training-time Z-rotation augmentation on the predicted orientation.
        # The policy emits rot in the rotated training frame; rotate by
        # +undo_z_rotation_deg around world-z to bring it back to the world frame.
        import pytorch3d.transforms as p3d
        R_pred_train = p3d.rotation_6d_to_matrix(rot6d_train.unsqueeze(0)).squeeze(0)
        R_pred = R_pred_train
        if self.config.undo_z_rotation_deg != 0.0:
            a = np.deg2rad(self.config.undo_z_rotation_deg)
            R_z = torch.tensor(
                [[np.cos(a), -np.sin(a), 0.0],
                 [np.sin(a),  np.cos(a), 0.0],
                 [0.0,        0.0,       1.0]], dtype=torch.float32, device=R_pred.device,
            )
            R_pred = R_z @ R_pred_train
        rot6d = p3d.matrix_to_rotation_6d(R_pred.unsqueeze(0)).squeeze(0)

        # print(
        #     f"[action] rot6d_train={rot6d_train.numpy().round(4)}  "
        #     f"rot6d_world={rot6d.numpy().round(4)}  "
        #     f"undo={self.config.undo_z_rotation_deg}°"
        # )

        # polaris → lerobot gripper convention flip
        gripper_lerobot = 1.0 - gripper_polaris

        # print(f"[action_eef] trans={pos.numpy().round(4)}  gripper={gripper_polaris.item():.4f}")

        if self._droid_adapter is not None:
            eef_lerobot = torch.cat([rot6d, pos, gripper_lerobot])
            state = batch.get("observation.state", torch.zeros(1, 8)).squeeze(0).cpu()
            joint_action = self._droid_adapter._eef_to_joints(eef_lerobot, state)
            action = joint_action.unsqueeze(0)
        else:
            # Fall back to polaris-format eef action (pos|rot6d|gripper)
            action = torch.cat([pos, rot6d, gripper_polaris]).unsqueeze(0)

        action_eef_lerobot = torch.cat([rot6d, pos, gripper_lerobot])
        return action, action_eef_lerobot.unsqueeze(0)

    def _extract_obs(self, batch: dict[str, Tensor]) -> dict:
        """Build the obs dict expected by ghost_server from a lerobot batch.

        """
        if len(self.config.cam_image_keys) != 3:
            raise ValueError(
                f"[GhostClient] expected exactly 3 cam_image_keys, got "
                f"{len(self.config.cam_image_keys)}"
            )

        obs: dict = {}

        # --- Images: preserve the original lerobot batch keys
        # (e.g. "observation.images.cam_azure_kinect_front.color"). The server reads
        # them via the `cam_keys` field in the request — see select_action.
        for cam_key in self.config.cam_image_keys:
            if cam_key not in batch:
                raise KeyError(
                    f"[GhostClient] cam_image_key '{cam_key}' not in batch. "
                    f"Available: {list(batch.keys())}"
                )
            val = batch[cam_key]  # (1, 3, H, W) float [0,1]
            if self.config.image_resize is not None:
                h, w = self.config.image_resize
                val = F.interpolate(val, size=(h, w), mode="bilinear", align_corners=False)
            img = val.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3) float
            obs[cam_key] = (img * 255.0).clip(0, 255).astype(np.uint8)

        # --- eef components from lerobot eef_pose [rot6d(6) | trans(3) | gripper(1)] ---
        if self.config.eef_pose_key not in batch:
            raise KeyError(
                f"[GhostClient] eef_pose_key '{self.config.eef_pose_key}' "
                f"not in batch. Available: {list(batch.keys())}"
            )
        eef = batch[self.config.eef_pose_key].squeeze(0)  # (10,)
        rot6d_lerobot   = eef[0:6]
        trans           = eef[6:9]
        gripper_lerobot = eef[9:10]  # lerobot: 0=closed, 1=open

        # --- Build R_obs (training-frame rotation matrix used by both states_ee and gripper_pcd) ---
        import pytorch3d.transforms as p3d
        R_obs = p3d.rotation_6d_to_matrix(rot6d_lerobot.unsqueeze(0)).squeeze(0)

        # Apply inverse Z rotation to match training augmentation
        # (mirrors transform_eef_dataset.py: R_train = R_z(-45°) @ R_world).
        if self.config.undo_z_rotation_deg != 0.0:
            a = np.deg2rad(-self.config.undo_z_rotation_deg)
            R_z = torch.tensor(
                [[np.cos(a), -np.sin(a), 0.0],
                 [np.sin(a),  np.cos(a), 0.0],
                 [0.0,        0.0,       1.0]], dtype=torch.float32, device=R_obs.device
            )
            R_obs = R_z @ R_obs

        # Continuous gripper (no threshold). lerobot 0=closed, 1=open → polaris 0=open, 1=closed.
        gripper_polaris = 1.0 - gripper_lerobot

        # --- gripper_pcd: matches eef_pose_to_gripper_pcd from create_pp_dataset_realrobot.py ---
        # Uses the SAME rotated R_obs and binarized gripper as training preprocessing.
        # Shape (4, 3) matches the input_feature spec [-1, 3] (no batch dim) AND
        # the high_level wrapper's expected layout (points axis at 0).
        obs["observation.points.gripper_pcds"] = self._synthesize_gripper_pcd(R_obs, trans, gripper_polaris)[[1, 2, 0, 3]]

        # --- eef_pose: (1, 10) [pos(3) | rot6d(6) | gripper_polaris(1)] (polaris layout)
        # R_obs already has the -45° z-aug applied so this is in the rotated training
        # frame; gripper is polaris convention (0=open, 1=closed). The ghost server
        # doesn't read this key (depth preprocessing only), so a single 10-d layout
        # works for both the round-trip through the server and the diffusion policy's
        # size-10 normalization stats.
        rot6d_polaris = p3d.matrix_to_rotation_6d(R_obs.unsqueeze(0)).squeeze(0)
        eef_pose_polaris = torch.cat(
            [trans, rot6d_polaris, gripper_polaris]
        ).unsqueeze(0)  # (1, 10) torch
        print(
            f"[obs] trans={trans.cpu().numpy().round(4)}  "
            f"rot6d={rot6d_polaris.cpu().numpy().round(4)}  "
            f"gripper_polaris={gripper_polaris.item():.1f}"
        )
        # obs is msgpack-serialized for the wire to ghost_server, which only handles
        # plain numpy. Store as float32 numpy here and re-tensorize in select_action
        # when overlaying back into batch_obs.
        obs["observation.right_eef_pose"] = eef_pose_polaris.cpu().numpy().astype(np.float32)  # (1, 10)

        return obs

    # Finger geometry constants — must match training preprocessing in
    # polaris/.../robot_utils/franka/create_pp_dataset_realrobot.py.
    _FINGER_OPEN_Y   = 0.05  # half-width when fully open (m)
    _FINGER_CLOSED_Y = 0.0   # half-width when fully closed (m)

    @classmethod
    def _synthesize_gripper_pcd(
        cls, R_obs: Tensor, trans: Tensor, gripper_polaris: Tensor
    ) -> np.ndarray:
        """Build a (4, 3) gripper_pcd matching eef_pose_to_gripper_pcd in
        create_pp_dataset_realrobot.py — local-frame offsets rotated to world.

        Args:
            R_obs: (3, 3) pytorch3d-row-convention EE rotation matrix in the same
                training frame used by the server (i.e. AFTER the -45° z-augmentation).
            trans: (3,) world-frame EE translation.
            gripper_polaris: (1,) continuous polaris gripper in [0, 1] (0=open, 1=closed).
                Width interpolates linearly: gw = OPEN*(1-gp) + CLOSED*gp.

        Layout (in EE local frame, then rotated to world via offsets @ R_obs + trans):
            [0] top    : [0, 0, -0.05]   — 5 cm along local -z
            [1] right  : [0, +gw, 0]     — along local +y by gw
            [2] left   : [0, -gw, 0]     — along local -y by gw
            [3] grasp  : [0, 0, 0]       — at EE origin
        where gw = FINGER_OPEN_Y if open else FINGER_CLOSED_Y.
        """
        gp = float(gripper_polaris.item())  # 0=open, 1=closed (already binarized)
        gw = cls._FINGER_OPEN_Y * (1.0 - gp) + cls._FINGER_CLOSED_Y * gp

        offsets = np.array(
            [
                [0.0,  0.0, -0.05],  # top
                [0.0,  gw,   0.0 ],  # right finger
                [0.0, -gw,   0.0 ],  # left finger
                [0.0,  0.0,  0.0 ],  # grasp center
            ],
            dtype=np.float32,
        )  # (4, 3) local frame

        # offsets @ R_pytorch3d gives world-frame coords (training does the same;
        # see create_pp_dataset_realrobot.py: rot_t = rot_mats.transpose then
        # pcd_world = offsets @ rot_t, where the double transpose cancels back to
        # pytorch3d's native row convention).
        R_np    = R_obs.detach().cpu().numpy().astype(np.float32)        # (3, 3)
        trans_np = trans.detach().cpu().numpy().astype(np.float32)        # (3,)
        pcd_world = offsets @ R_np + trans_np                             # (4, 3)
        return pcd_world                                                  # (4, 3)

    def shutdown(self):
        """Send shutdown signal to the server."""
        try:
            asyncio.run(self._send({"command": "shutdown"}))
            print(f"[GhostClient] Shutdown signal sent to {self.uri}")
        except Exception as e:
            print(f"[GhostClient] Server {self.uri} closed: {e}")

    # --- Required abstract methods ---

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("GhostClient is inference-only; use select_action().")
