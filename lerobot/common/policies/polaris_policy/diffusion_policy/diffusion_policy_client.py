"""
DiffusionPolicyClient: a thin ZMQ client wrapper around polaris_server.py.

The policy has no local model weights. It forwards preprocessed observations
to the polaris_server (running in the robodiff conda env) via ZMQ and returns
the action tensors it receives back.

Integration with control_robot.py:
    python lerobot/scripts/control_robot.py \\
        --robot.type=droid \\
        --control.type=record \\
        --control.policy.type=diffusion_policy \\
        --control.policy.host=localhost \\
        --control.policy.port=5557 \\
        --control.policy.obs_key_map='{"observation.images.cam_azure_kinect_left.color": "image",
                                        "observation.right_eef_pose": "agent_pos"}' \\
        --control.policy.image_resize='[240, 426]' \\
        --control.policy.transform_eef_to_agent_pos=true \\
        ...

obs_key_map maps lerobot observation keys to the keys expected by the
polaris diffusion policy (as defined in its training shape_meta).

Per-key transforms applied before sending to the server:
  - image keys: resized to image_resize (H, W) if set
  - agent_pos key: if transform_eef_to_agent_pos=True, reorders
      lerobot eef_pose [rot6d(6) | trans(3) | gripper(1)]
    to polaris agent_pos [trans(3) | rot6d(6) | gripper_binary(1)]
    where gripper_binary = 1.0 if gripper < gripper_binarize_threshold else 0.0
    (threshold defaults to 0.8, configurable via --control.policy.gripper_binarize_threshold)
"""

import pickle
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from torch import Tensor


def _enc(v):
    if isinstance(v, np.ndarray):
        return {"__ndarray__": True, "dtype": str(v.dtype),
                "shape": list(v.shape), "data": v.tobytes()}
    if isinstance(v, dict):
        return {k2: _enc(v2) for k2, v2 in v.items()}
    return v


def _dec(v):
    if isinstance(v, dict) and v.get("__ndarray__") is True:
        return np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"]).copy()
    if isinstance(v, dict):
        return {k2: _dec(v2) for k2, v2 in v.items()}
    return v


def _serialize(obj) -> bytes:
    return pickle.dumps(_enc(obj))


def _deserialize(data: bytes):
    return _dec(pickle.loads(data))

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("diffusion_policy")
@dataclass
class DiffusionPolicyConfig(PreTrainedConfig):
    """Configuration for DiffusionPolicyClient (ZMQ client to polaris_server.py).

    Args:
        host: Hostname or IP of the machine running polaris_server.py.
        port: ZMQ port that polaris_server.py is listening on.
        obs_key_map: Maps lerobot observation keys to polaris policy keys.
            Only keys listed here are sent; all others are dropped.
            If empty, all non-task keys are forwarded with their original names.
        image_resize: (H, W) to resize all image observations before sending.
            e.g. [240, 426] to match polaris diffusion policy training resolution.
            If None, images are sent at their original resolution.
        transform_eef_to_agent_pos: If True, the key mapped to "agent_pos" is
            transformed from lerobot eef_pose format [rot6d(6)|trans(3)|gripper(1)]
            to polaris agent_pos format [trans(3)|rot6d(6)|gripper_binary(1)],
            where gripper_binary = 1.0 if gripper < gripper_binarize_threshold else 0.0.
        gripper_binarize_threshold: Width threshold (in lerobot gripper units) used
            when binarizing the agent_pos gripper. Widths < threshold → 1.0 (closed
            in polaris convention); widths ≥ threshold → 0.0 (open). Default 0.8.
    """

    host: str = "localhost"
    port: int = 5557
    obs_key_map: dict = field(default_factory=dict)
    image_resize: list | None = None          # [H, W], e.g. [240, 426]
    transform_eef_to_agent_pos: bool = False
    gripper_binarize_threshold: float = 0.8   # width <threshold → 1.0 (closed in polaris); ≥threshold → 0.0 (open)
    use_ik: bool = False                      # convert EEF action → joint positions via deoxys IKWrapper
    enable_goal_conditioning: bool = False    # compatibility with control_robot.py
    undo_z_rotation_deg: float = 0.0         # undo training-time Z rotation; e.g. 45.0 to undo -45° aug

    # DiffusionPolicyClient has no local model — these are dummies for interface compatibility
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


class DiffusionPolicyClient(PreTrainedPolicy):
    """Thin ZMQ client policy that forwards observations to polaris_server.py."""

    config_class = DiffusionPolicyConfig
    name = "diffusion_policy"

    def __init__(self, config: DiffusionPolicyConfig, **kwargs):
        super().__init__(config)
        self.config = config

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{config.host}:{config.port}")
        print(f"[DiffusionPolicyClient] Connected to polaris server at {config.host}:{config.port}")

        if config.use_ik:
            from lerobot.common.policies.robot_adapters import DroidAdapter
            self._droid_adapter = DroidAdapter(action_space="right_eef")
        else:
            self._droid_adapter = None

        # Per-episode gripper trace buffers. agent_pos: lerobot width (before) and
        # polaris binary (after). action: polaris binary from server (before) and
        # lerobot convention 1-x (after).
        import time as _time, atexit as _atexit
        self._gripper_buf = {
            "agent_pos_before": [],
            "agent_pos_after": [],
            "action_before": [],
            "action_after": [],
        }
        self._episode_idx = 0
        self._gripper_plot_dir = (
            f"outputs/inference_gripper_plots/{_time.strftime('%Y%m%d_%H%M%S')}"
        )
        _atexit.register(self._flush_gripper_plot)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path=None, *, config=None, **kwargs):
        """Override: DiffusionPolicyClient has no model weights to load."""
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy = cls(config)
        policy.eval()
        return policy

    def reset(self):
        """Signal the server to reset its observation/action buffers."""
        self._flush_gripper_plot()
        self.socket.send(_serialize({"reset": True}))
        self.socket.recv()

    def _flush_gripper_plot(self):
        """Save the current episode's gripper trace and clear buffers."""
        n = len(self._gripper_buf["action_before"])
        if n == 0 and len(self._gripper_buf["agent_pos_before"]) == 0:
            return
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self._gripper_plot_dir, exist_ok=True)
        path = os.path.join(self._gripper_plot_dir, f"episode_{self._episode_idx:03d}.png")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax = axes[0]
        ax.plot(self._gripper_buf["agent_pos_before"],
                label="before transform (lerobot width)", color="C0")
        ax.plot(self._gripper_buf["agent_pos_after"],
                label="after transform (polaris binary, 1=closed)",
                color="C1", linestyle="--")
        ax.set_ylabel("agent_pos[-1]")
        ax.set_title(f"Episode {self._episode_idx} — gripper trace")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(self._gripper_buf["action_before"],
                label="before transform (polaris binary from server, 1=closed)",
                color="C2")
        ax.plot(self._gripper_buf["action_after"],
                label="after transform (lerobot, 1=open)",
                color="C3", linestyle="--")
        ax.set_ylabel("action[-1]")
        ax.set_xlabel("inference step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[DiffusionPolicyClient] Saved gripper plot: {path} "
              f"({n} action steps, {len(self._gripper_buf['agent_pos_before'])} obs steps)")

        for k in self._gripper_buf:
            self._gripper_buf[k].clear()
        self._episode_idx += 1

    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Send current observation to polaris_server and return (action, action_eef).

        Args:
            batch: Preprocessed observation dict from predict_action().
                   Images are (1, 3, H, W) float32 [0,1] on device.
                   State/eef tensors are (1, D) float32 on device.
                   task is list[str].

        Returns:
            action: (1, action_dim) float32 tensor.
            action_eef: (1, action_dim) float32 tensor (same as action).
        """
        obs_np = self._batch_to_numpy(batch)
        task = batch.get("task", [""])
        task_str = task[0] if isinstance(task, list) else task

        request = {"obs": obs_np, "task": task_str}
        self.socket.send(_serialize(request))

        response = _deserialize(self.socket.recv())
        if "error" in response:
            raise RuntimeError(f"[DiffusionPolicyClient] Server error: {response['error']}")

        action_eef_np = response["action_eef"]  # (10,) [trans(3)|rot6d(6)|gripper(1)]
        action_eef = torch.from_numpy(action_eef_np).float()

        # Undo training-time Z-rotation augmentation if requested
        if self.config.undo_z_rotation_deg != 0.0:
            import pytorch3d.transforms as transforms
            a = np.deg2rad(self.config.undo_z_rotation_deg)
            rot6d = action_eef[3:9]
            R_z = torch.tensor(
                [[np.cos(a), -np.sin(a), 0.0],
                 [np.sin(a),  np.cos(a), 0.0],
                 [0.0,        0.0,       1.0]], dtype=torch.float32, device=rot6d.device
            )
            R_pred = transforms.rotation_6d_to_matrix(rot6d.unsqueeze(0)).squeeze(0)
            R_undone = R_z @ R_pred
            rot6d_new = transforms.matrix_to_rotation_6d(R_undone.unsqueeze(0)).squeeze(0)
            action_eef = torch.cat([action_eef[0:3], rot6d_new, action_eef[9:10]])

        # Server returns [trans(3)|rot6d(6)|gripper(1)]
        trans   = action_eef[0:3]
        rot6d   = action_eef[3:9]
        gripper = action_eef[9:10]
        print(f"[action_eef] trans={trans.numpy().round(4)}  gripper={gripper.item():.4f}")

        self._gripper_buf["action_before"].append(float(gripper.item()))
        self._gripper_buf["action_after"].append(float((1.0 - gripper).item()))

        if self._droid_adapter is not None:
            # Server returns binary gripper in polaris convention (1=closed, 0=open).
            # Flip to lerobot convention (0=closed, 1=open).
            gripper_lerobot = 1.0 - gripper
            eef_lerobot = torch.cat([rot6d, trans, gripper_lerobot])  # lerobot format
            state = batch.get("observation.state", torch.zeros(1, 8)).squeeze(0).cpu()
            joint_action = self._droid_adapter._eef_to_joints(eef_lerobot, state)
            action = joint_action.unsqueeze(0)
        else:
            action = action_eef.unsqueeze(0)

        # Return action_eef in lerobot format [rot6d(6)|trans(3)|gripper_lerobot(1)] for dataset recording
        gripper_lerobot = 1.0 - gripper
        action_eef_lerobot = torch.cat([rot6d, trans, gripper_lerobot])
        return action, action_eef_lerobot.unsqueeze(0)

    def _batch_to_numpy(self, batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
        """Convert the lerobot batch to a numpy dict for ZMQ serialization.

        Applies obs_key_map renaming, per-key transforms, and strips batch dim.

        Image tensors arrive as (1, 3, H, W) float32 [0,1].
        State/eef tensors arrive as (1, D) float32.
        The server receives (3, H, W) and (D,) respectively (no batch dim).
        """
        key_map = self.config.obs_key_map
        obs_np = {}

        # Find which lerobot key holds eef_pose (source-based, policy-name-agnostic)
        agent_pos_lerobot_key = None
        if self.config.transform_eef_to_agent_pos and key_map:
            for lk in key_map:
                if "eef_pose" in lk or "ee_pose" in lk:
                    agent_pos_lerobot_key = lk
                    break

        for lerobot_key, val in batch.items():
            if lerobot_key == "task":
                continue
            if not isinstance(val, Tensor):
                continue

            # Determine the server-side key name
            if key_map:
                if lerobot_key not in key_map:
                    continue  # drop keys not in map
                server_key = key_map[lerobot_key]
            else:
                server_key = lerobot_key

            # --- Per-key transforms ---

            # Image resize: val is (1, 3, H, W) float [0,1]
            if "image" in lerobot_key and self.config.image_resize is not None:
                h, w = self.config.image_resize
                val = F.interpolate(val, size=(h, w), mode="bilinear", align_corners=False)

            # eef_pose → agent_pos reorder + gripper binarization + optional Z rotation
            # lerobot eef_pose: [rot6d(0:6) | trans(6:9) | gripper(9)]  shape (1, 10)
            # polaris agent_pos: [trans(3) | rot6d(6) | gripper_binary(1)]  shape (1, 10)
            if lerobot_key == agent_pos_lerobot_key:
                val = val.squeeze(0)  # (10,)
                rot6d   = val[0:6]
                trans   = val[6:9]
                gripper = val[9:10]

                # Apply the same Z rotation used during training so obs matches training distribution.
                # undo_z_rotation_deg undoes it on the action side, so we apply the inverse here.
                if self.config.undo_z_rotation_deg != 0.0:
                    import pytorch3d.transforms as transforms
                    a = np.deg2rad(-self.config.undo_z_rotation_deg)
                    R_z = torch.tensor(
                        [[np.cos(a), -np.sin(a), 0.0],
                         [np.sin(a),  np.cos(a), 0.0],
                         [0.0,        0.0,       1.0]], dtype=torch.float32, device=rot6d.device
                    )
                    R_obs = transforms.rotation_6d_to_matrix(rot6d.unsqueeze(0)).squeeze(0)
                    rot6d = transforms.matrix_to_rotation_6d((R_z @ R_obs).unsqueeze(0)).squeeze(0)

                gripper_binary = torch.where(gripper < self.config.gripper_binarize_threshold,
                                             torch.ones_like(gripper),
                                             torch.zeros_like(gripper))
                self._gripper_buf["agent_pos_before"].append(float(gripper.item()))
                self._gripper_buf["agent_pos_after"].append(float(gripper_binary.item()))
                val = torch.cat([trans, rot6d, gripper_binary], dim=0).unsqueeze(0)  # (1, 10)

            arr = val.squeeze(0).cpu().numpy()  # remove batch dim
            obs_np[server_key] = arr

        return obs_np

    # --- Required abstract methods (unused for inference-only policy) ---

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("DiffusionPolicyClient is inference-only; use select_action().")
