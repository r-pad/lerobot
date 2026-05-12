"""
AMPLIFYPolicyClient: ZMQ client wrapper around amplify_server.py.

The policy has no local model weights. It forwards preprocessed observations
to the amplify_server (running in the amplify conda env) via ZMQ and returns
the action tensors it receives back.

Integration with control_robot.py:
    python lerobot/scripts/control_robot.py \\
        --robot.type=droid \\
        --control.type=record \\
        --control.policy.type=amplify \\
        --control.policy.host=localhost \\
        --control.policy.port=5558 \\
        --control.policy.cam_image_keys='["observation.images.cam_azure_kinect_front.color",
                                          "observation.images.cam_azure_kinect_left.color"]' \\
        --control.policy.image_resize='[240, 426]' \\
        --control.policy.transform_eef_to_agent_pos=true \\
        --control.policy.use_ik=true \\
        --control.policy.open_loop_horizon=1 \\
        --control.policy.obs_key_map='{"observation.right_eef_pose": "proprio"}' \\
        ...

Server request format:
    {"image": np.ndarray (v, H, W, 3) float32 [0,1],
     "proprio": np.ndarray (10,) float32 = pos(3)+rot6d(6)+gripper(1)}
    or {"reset": True}

Server response format:
    {"action_chunk": np.ndarray (action_horizon, 10) float32}
    actions are pos(3)+rot6d(6)+gripper(1) in polaris convention (gripper: 1=closed, 0=open).
"""

import pickle
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from torch import Tensor

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("amplify")
@dataclass
class AMPLIFYPolicyConfig(PreTrainedConfig):
    """Configuration for AMPLIFYPolicyClient (ZMQ client to amplify_server.py).

    Args:
        host: Hostname of the machine running amplify_server.py.
        port: ZMQ port that amplify_server.py is listening on.
        cam_image_keys: Ordered list of lerobot camera observation keys to use as
            AMPLIFY views [view0, view1, ...]. Stacked in this order as (v, H, W, 3).
        obs_key_map: Maps lerobot observation keys to server keys.
            Only non-image keys need to be listed (images are handled via cam_image_keys).
            Typically: {"observation.right_eef_pose": "proprio"}.
        image_resize: (H, W) to resize camera images before sending.
        transform_eef_to_agent_pos: If True, reorders lerobot eef_pose
            [rot6d(6)|trans(3)|gripper(1)] to polaris proprio
            [trans(3)|rot6d(6)|gripper_binary(1)], where the gripper is flipped
            from lerobot (0=closed, 1=open) to polaris (1=closed, 0=open) and
            binarized: lerobot < gripper_binarize_threshold → 1.0, else 0.0.
        gripper_binarize_threshold: Width threshold (in lerobot gripper units) used
            when binarizing the proprio gripper. Widths < threshold → 1.0 (closed
            in polaris convention); widths ≥ threshold → 0.0 (open). Default 0.5.
        use_ik: If True, convert EEF action → joint positions via DroidAdapter IK.
        open_loop_horizon: Steps to execute from each action chunk before re-querying.
        undo_z_rotation_deg: Undo training-time Z rotation on actions and obs.
    """

    host: str = "localhost"
    port: int = 5558
    cam_image_keys: list = field(default_factory=list)
    obs_key_map: dict = field(default_factory=dict)
    image_resize: list | None = None
    transform_eef_to_agent_pos: bool = False
    gripper_binarize_threshold: float = 0.5   # width <threshold → 1.0 (closed in polaris); ≥threshold → 0.0 (open)
    use_ik: bool = False
    open_loop_horizon: int = 1
    undo_z_rotation_deg: float = 0.0
    vis_tracks: bool = False  # save predicted track visualization into the dataset (requires --vis_tracks on server)
    # Observation key under which `_current_vis_frame` is recorded. Each policy
    # that emits a per-step visualization should set this to a unique value so
    # the global recording loop in control_utils.py can store it without naming
    # collisions across policies.
    vis_obs_key: str = "observation.images.amplify_tracks"
    enable_goal_conditioning: bool = False  # compatibility with control_robot.py
    use_amp: bool = False

    @property
    def vis_shape(self) -> tuple[int, int, int] | None:
        """(H, W*num_views, 3) shape of the per-step track visualization frame.

        Returns None when vis_tracks is off or the inputs needed to compute the
        shape (image_resize, cam_image_keys) aren't provided, so control_robot.py
        skips registering the dataset feature.
        """
        if not self.vis_tracks or self.image_resize is None or not self.cam_image_keys:
            return None
        h, w = self.image_resize
        return (h, w * len(self.cam_image_keys), 3)

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


class AMPLIFYPolicyClient(PreTrainedPolicy):
    """Thin ZMQ client policy that forwards observations to amplify_server.py."""

    config_class = AMPLIFYPolicyConfig
    name = "amplify"

    def __init__(self, config: AMPLIFYPolicyConfig, **kwargs):
        super().__init__(config)
        self.config = config

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{config.host}:{config.port}")
        print(f"[AMPLIFYPolicyClient] Connected to AMPLIFY server at {config.host}:{config.port}")
        print(f"[AMPLIFYPolicyClient] Camera views: {config.cam_image_keys}")
        print(f"[AMPLIFYPolicyClient] Open-loop horizon: {config.open_loop_horizon}")

        if config.use_ik:
            from lerobot.common.policies.robot_adapters import DroidAdapter
            self._droid_adapter = DroidAdapter(action_space="right_eef")
        else:
            self._droid_adapter = None

        self._action_chunk: np.ndarray | None = None
        self._actions_done = 0
        self._current_vis_frame: np.ndarray | None = None  # (H, W*v, 3) uint8 RGB, updated each inference

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path=None, *, config=None, **kwargs):
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy = cls(config)
        policy.eval()
        return policy

    def reset(self):
        self._action_chunk = None
        self._actions_done = 0
        self._current_vis_frame = None
        self.socket.send(pickle.dumps({"reset": True}))
        self.socket.recv()

    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Send observation to amplify_server and return (action, action_eef).

        Returns:
            action: (1, action_dim) joint positions or EEF depending on use_ik.
            action_eef: (1, 10) in lerobot EEF format [rot6d(6)|trans(3)|gripper_lerobot(1)].
        """
        need_inference = (
            self._action_chunk is None
            or self._actions_done >= self.config.open_loop_horizon
        )

        if need_inference:
            image, proprio = self._extract_obs(batch)
            request = {"image": image, "proprio": proprio}
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            if "error" in response:
                raise RuntimeError(f"[AMPLIFYPolicyClient] Server error: {response['error']}")
            self._action_chunk = response["action_chunk"]  # (action_horizon, 10)
            self._actions_done = 0

            if "vis_frame" in response and self.config.vis_tracks:
                self._current_vis_frame = response["vis_frame"]  # (H, W*v, 3) uint8 RGB

        # action_chunk[i] is polaris format: [trans(3)|rot6d(6)|gripper_polaris(1)]
        action10_np = self._action_chunk[self._actions_done]  # (10,)
        self._actions_done += 1

        action_eef = torch.from_numpy(action10_np).float()  # [trans(3)|rot6d(6)|gripper_polaris]

        # Undo Z-rotation augmentation on action
        if self.config.undo_z_rotation_deg != 0.0:
            import pytorch3d.transforms as p3d
            a = np.deg2rad(self.config.undo_z_rotation_deg)
            rot6d = action_eef[3:9]
            R_z = torch.tensor(
                [[np.cos(a), -np.sin(a), 0.0],
                 [np.sin(a),  np.cos(a), 0.0],
                 [0.0,        0.0,       1.0]], dtype=torch.float32, device=rot6d.device
            )
            R_pred = p3d.rotation_6d_to_matrix(rot6d.unsqueeze(0)).squeeze(0)
            rot6d_new = p3d.matrix_to_rotation_6d((R_z @ R_pred).unsqueeze(0)).squeeze(0)
            action_eef = torch.cat([action_eef[0:3], rot6d_new, action_eef[9:10]])

        trans   = action_eef[0:3]
        rot6d   = action_eef[3:9]
        gripper = action_eef[9:10]  # polaris: 1=closed, 0=open
        print(f"[action_eef] trans={trans.numpy().round(4)}  gripper={gripper.item():.4f}")

        gripper_lerobot = 1.0 - gripper  # lerobot: 0=closed, 1=open

        if self._droid_adapter is not None:
            eef_lerobot = torch.cat([rot6d, trans, gripper_lerobot])  # lerobot format
            state = batch.get("observation.state", torch.zeros(1, 8)).squeeze(0).cpu()
            joint_action = self._droid_adapter._eef_to_joints(eef_lerobot, state)
            action = joint_action.unsqueeze(0)
        else:
            action = action_eef.unsqueeze(0)

        # action_eef for dataset recording: lerobot format [rot6d(6)|trans(3)|gripper_lerobot(1)]
        action_eef_lerobot = torch.cat([rot6d, trans, gripper_lerobot])
        return action, action_eef_lerobot.unsqueeze(0)

    def _extract_obs(self, batch: dict[str, Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """Extract (image, proprio) from lerobot batch for the AMPLIFY server.

        Returns:
            image:   (v, H, W, 3) float32 [0,1]
            proprio: (10,) float32 = pos(3)+rot6d(6)+gripper(1)  (polaris format)
        """
        # --- Images: stack camera views in order ---
        images = []
        for cam_key in self.config.cam_image_keys:
            if cam_key not in batch:
                raise KeyError(f"[AMPLIFYPolicyClient] cam_image_key '{cam_key}' not in batch. "
                               f"Available: {list(batch.keys())}")
            val = batch[cam_key]  # (1, 3, H, W) float [0,1]
            if self.config.image_resize is not None:
                h, w = self.config.image_resize
                val = F.interpolate(val, size=(h, w), mode="bilinear", align_corners=False)
            # (1, 3, H, W) → (H, W, 3)
            img_hwc = val.squeeze(0).permute(1, 2, 0).cpu().numpy()
            images.append(img_hwc)
        image = np.stack(images, axis=0).astype(np.float32)  # (v, H, W, 3)

        # --- Proprio: find eef_pose key and transform ---
        key_map = self.config.obs_key_map
        proprio = None

        # Find which lerobot key is the eef pose (source-based)
        eef_lerobot_key = None
        if self.config.transform_eef_to_agent_pos and key_map:
            for lk in key_map:
                if "eef_pose" in lk or "ee_pose" in lk:
                    eef_lerobot_key = lk
                    break

        if eef_lerobot_key is not None and eef_lerobot_key in batch:
            val = batch[eef_lerobot_key].squeeze(0)  # (10,) lerobot: [rot6d(6)|trans(3)|gripper(1)]
            rot6d   = val[0:6]
            trans   = val[6:9]
            gripper = val[9:10]

            # Apply inverse Z rotation to obs (same magnitude as action undo, but inverse direction)
            if self.config.undo_z_rotation_deg != 0.0:
                import pytorch3d.transforms as p3d
                a = np.deg2rad(-self.config.undo_z_rotation_deg)
                R_z = torch.tensor(
                    [[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a),  np.cos(a), 0.0],
                     [0.0,        0.0,       1.0]], dtype=torch.float32, device=rot6d.device
                )
                R_obs = p3d.rotation_6d_to_matrix(rot6d.unsqueeze(0)).squeeze(0)
                rot6d = p3d.matrix_to_rotation_6d((R_z @ R_obs).unsqueeze(0)).squeeze(0)

            # Flip lerobot (0=closed, 1=open) → polaris (1=closed, 0=open) and binarize
            # in a single step: lerobot < threshold → 1.0 (closed), else 0.0 (open).
            gripper = torch.where(gripper < self.config.gripper_binarize_threshold,
                                  torch.ones_like(gripper),
                                  torch.zeros_like(gripper))

            # polaris proprio format: [trans(3)|rot6d(6)|gripper_binary_polaris(1)]
            proprio = torch.cat([trans, rot6d, gripper], dim=0).cpu().numpy().astype(np.float32)

        elif key_map:
            # Fall back: look for any non-image key in obs_key_map
            for lk, sk in key_map.items():
                if "image" not in lk and lk in batch:
                    proprio = batch[lk].squeeze(0).cpu().numpy().astype(np.float32)
                    break

        if proprio is None:
            raise ValueError(
                "[AMPLIFYPolicyClient] Could not find proprio in batch. "
                "Set obs_key_map and/or transform_eef_to_agent_pos."
            )

        return image, proprio

    # --- Required abstract methods ---

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("AMPLIFYPolicyClient is inference-only; use select_action().")
