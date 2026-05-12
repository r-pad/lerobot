"""
PointPolicyClient: a thin websockets/msgpack client wrapper around point_policy_server.py.

The policy has no local model weights. It forwards preprocessed observations
to point_policy_server (running in the point-policy conda env) over websockets
and returns the action it receives back.

Integration with control_robot.py:
    python lerobot/scripts/control_robot.py \\
        --robot.type=droid \\
        --control.type=record \\
        --control.policy.type=point \\
        --control.policy.host=localhost \\
        --control.policy.port=8765 \\
        --control.policy.cam_image_keys='["observation.images.cam_azure_kinect_left.color",
                                          "observation.images.cam_azure_kinect_front.color"]' \\
        --control.policy.image_resize='[240, 426]' \\
        --control.policy.eef_pose_key=observation.right_eef_pose \\
        --control.policy.transform_eef_to_states_ee=true \\
        --control.policy.use_ik=true \\
        ...

Server protocol (websockets + msgpack with msgpack_numpy):
    request:  {"command": "infer",
               "obs": {"pixels1": (H,W,3) uint8,
                       "pixels2": (H,W,3) uint8,
                       "gripper_pcd": (1, 4, 3) float32 — synthesized from eef_pose,
                       "states_ee": (1, 8) float32 — pos(3)|quat_wxyz(4)|gripper(1)},
               "return_viz": bool}
    response: {"action": (8,) float32 — pos(3)|quat_wxyz(4)|gripper(1) (0=open, 1=closed),
               "viz":    (H, W*v, 3) uint8 | None}

    request:  {"command": "reset"}
    response: {"status": "reset"}

    request:  {"command": "shutdown"}
    response: {"status": "shutdown"}
"""

import asyncio
from dataclasses import dataclass, field

import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np
import torch
import torch.nn.functional as F
import websockets
from torch import Tensor

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("point-policy")
@dataclass
class PointPolicyConfig(PreTrainedConfig):
    """Configuration for PointPolicyClient (websockets client to point_policy_server.py).

    Args:
        host: Hostname of the machine running point_policy_server.py.
        port: websockets port that point_policy_server.py is listening on.
        cam_image_keys: Ordered list of two lerobot camera observation keys to use as
            point-policy views [pixels1, pixels2].
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
    """

    host: str = "localhost"
    port: int = 8766
    cam_image_keys: list = field(default_factory=list)
    vis_obs_key: str = "observation.images.point_policy_vis"
    # (H, W*num_views, 3) shape of the per-step server viz frame. Default matches
    # the server's two-view 720x1280 native output. Override on the CLI if your
    # server emits a different size.
    vis_shape: list = field(default_factory=lambda: [720, 1280, 3])
    eef_pose_key: str = "observation.right_eef_pose"
    image_resize: list | None = None
    # Server expects polaris-format states_ee [trans(3)|quat_wxyz(4)|gripper(1)] (indexed
    # at [3:7] for quat). lerobot format is 10-d rot6d+trans+gripper, so this conversion
    # is required for the server to read the obs correctly.
    transform_eef_to_states_ee: bool = True
    use_ik: bool = False
    # Dataset was preprocessed with -45° z-axis world-frame rotation on the rotation part
    # (commit bb6730e). Default 45.0 undoes that on the action side and applies the inverse
    # (-45°) on the obs side so the policy sees rotations in the same frame as training.
    undo_z_rotation_deg: float = 45.0
    vis_tracks: bool = False
    enable_goal_conditioning: bool = False  # compatibility with control_robot.py

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


class PointPolicyClient(PreTrainedPolicy):
    """Thin websockets client that forwards observations to point_policy_server.py."""

    config_class = PointPolicyConfig
    name = "point_policy"

    def __init__(self, config: PointPolicyConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.uri = f"ws://{config.host}:{config.port}"
        print(f"[PointPolicyClient] Server URI: {self.uri}")
        print(f"[PointPolicyClient] Camera views: {config.cam_image_keys}")

        if config.use_ik:
            from lerobot.common.policies.robot_adapters import DroidAdapter
            self._droid_adapter = DroidAdapter(action_space="right_eef")
        else:
            self._droid_adapter = None

        self._current_vis_frame: np.ndarray | None = None  # (H, W*v, 3) uint8 RGB

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path=None, *, config=None, **kwargs):
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy = cls(config)
        policy.eval()
        return policy

    async def _send(self, message: dict) -> dict:
        async with websockets.connect(self.uri, max_size=100 * 1024 * 1024) as ws:
            await ws.send(msgpack.packb(message, use_bin_type=True))
            response = await ws.recv()
            return msgpack.unpackb(response, raw=False)

    def reset(self):
        self._current_vis_frame = None
        asyncio.run(self._send({"command": "reset"}))

    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Send observation to point_policy_server and return (action, action_eef).

        Returns:
            action: (1, action_dim) joint positions or EEF depending on use_ik.
            action_eef: (1, 10) lerobot EEF format [rot6d(6)|trans(3)|gripper_lerobot(1)].
        """
        obs = self._extract_obs(batch)
        result = asyncio.run(self._send({
            "command": "infer",
            "obs": obs,
            "return_viz": self.config.vis_tracks,
        }))
        if "error" in result:
            raise RuntimeError(f"[PointPolicyClient] Server error: {result['error']}")

        # Server action: (8,) pos(3) | quat_wxyz(4) | gripper(1)  (0=open, 1=closed)
        action_ee_np = np.asarray(result["action"]).astype(np.float32).reshape(-1)
        print(f"[PointPolicyClient] Received action: {action_ee_np}")
        if action_ee_np.shape[0] != 8:
            raise ValueError(f"Expected action shape (8,), got {action_ee_np.shape}")

        if result.get("viz") is not None and self.config.vis_tracks:
            self._current_vis_frame = np.asarray(result["viz"])

        pos = torch.from_numpy(action_ee_np[0:3])
        quat_wxyz = torch.from_numpy(action_ee_np[3:7])
        gripper_polaris = torch.from_numpy(action_ee_np[7:8])  # 0=open, 1=closed

        # Undo training-time Z-rotation augmentation on the predicted gripper orientation
        # (mirrors amplify_client.py select_action). Server returns rot in the rotated frame;
        # we rotate by +undo_z_rotation_deg around world-z to bring it back to world frame.
        import pytorch3d.transforms as p3d
        R_pred_train = p3d.quaternion_to_matrix(quat_wxyz.unsqueeze(0)).squeeze(0)
        R_pred = R_pred_train
        if self.config.undo_z_rotation_deg != 0.0:
            a = np.deg2rad(self.config.undo_z_rotation_deg)
            R_z = torch.tensor(
                [[np.cos(a), -np.sin(a), 0.0],
                 [np.sin(a),  np.cos(a), 0.0],
                 [0.0,        0.0,       1.0]], dtype=torch.float32, device=R_pred.device
            )
            R_pred = R_z @ R_pred_train
        rot6d = p3d.matrix_to_rotation_6d(R_pred.unsqueeze(0)).squeeze(0)

        # Debug: print quat (rotated frame, from server) and quat_world (after +undo).
        quat_world_wxyz = p3d.matrix_to_quaternion(R_pred.unsqueeze(0)).squeeze(0)
        print(
            f"[action] quat_train={quat_wxyz.numpy().round(4)}  "
            f"quat_world={quat_world_wxyz.numpy().round(4)}  "
            f"undo={self.config.undo_z_rotation_deg}°"
        )

        # polaris → lerobot gripper convention flip
        gripper_lerobot = 1.0 - gripper_polaris

        print(f"[action_eef] trans={pos.numpy().round(4)}  gripper={gripper_polaris.item():.4f}")

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
        """Build the obs dict expected by point_policy_server from a lerobot batch.

        Returns dict with keys: pixels1, pixels2, gripper_pcd, states_ee.
        """
        if len(self.config.cam_image_keys) != 2:
            raise ValueError(
                f"[PointPolicyClient] expected exactly 2 cam_image_keys, got "
                f"{len(self.config.cam_image_keys)}"
            )

        obs: dict = {}

        # --- Images: pixels1 / pixels2 as HxWx3 uint8 RGB ---
        pixel_keys = ["pixels1", "pixels2"]
        for cam_key, pkey in zip(self.config.cam_image_keys, pixel_keys):
            if cam_key not in batch:
                raise KeyError(
                    f"[PointPolicyClient] cam_image_key '{cam_key}' not in batch. "
                    f"Available: {list(batch.keys())}"
                )
            val = batch[cam_key]  # (1, 3, H, W) float [0,1]
            if self.config.image_resize is not None:
                h, w = self.config.image_resize
                val = F.interpolate(val, size=(h, w), mode="bilinear", align_corners=False)
            img = val.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3) float
            obs[pkey] = (img * 255.0).clip(0, 255).astype(np.uint8)

        # --- eef components from lerobot eef_pose [rot6d(6) | trans(3) | gripper(1)] ---
        if self.config.eef_pose_key not in batch:
            raise KeyError(
                f"[PointPolicyClient] eef_pose_key '{self.config.eef_pose_key}' "
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
        obs["gripper_pcd"] = self._synthesize_gripper_pcd(R_obs, trans, gripper_polaris)

        # --- states_ee: (1, 8) [pos(3) | quat_wxyz(4) | gripper(1)] (polaris) ---
        if self.config.transform_eef_to_states_ee:
            quat_wxyz = p3d.matrix_to_quaternion(R_obs.unsqueeze(0)).squeeze(0)
            states_ee = torch.cat([trans, quat_wxyz, gripper_polaris]).cpu().numpy().astype(np.float32)
            print(
                f"[obs] trans={trans.cpu().numpy().round(4)}  "
                f"quat_wxyz={quat_wxyz.cpu().numpy().round(4)}  "
                f"gripper_polaris={gripper_polaris.item():.1f}"
            )
        else:
            states_ee = eef.cpu().numpy().astype(np.float32)

        obs["states_ee"] = states_ee[None]  # (1, 8)

        return obs

    # Finger geometry constants — must match training preprocessing in
    # polaris/.../robot_utils/franka/create_pp_dataset_realrobot.py.
    _FINGER_OPEN_Y   = 0.05  # half-width when fully open (m)
    _FINGER_CLOSED_Y = 0.0   # half-width when fully closed (m)

    @classmethod
    def _synthesize_gripper_pcd(
        cls, R_obs: Tensor, trans: Tensor, gripper_polaris: Tensor
    ) -> np.ndarray:
        """Build a (1, 4, 3) gripper_pcd matching eef_pose_to_gripper_pcd in
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
        return pcd_world[None]                                            # (1, 4, 3)

    def shutdown(self):
        """Send shutdown signal to the server."""
        try:
            asyncio.run(self._send({"command": "shutdown"}))
            print(f"[PointPolicyClient] Shutdown signal sent to {self.uri}")
        except Exception as e:
            print(f"[PointPolicyClient] Server {self.uri} closed: {e}")

    # --- Required abstract methods ---

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("PointPolicyClient is inference-only; use select_action().")
