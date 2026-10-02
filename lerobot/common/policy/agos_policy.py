"""Real-robot deployment of an AGOS visuo-tactile diffusion policy (e.g. agos_canonical_split_unet).

The network, checkpoint loading, normalizers, rotation limiter and EEF->world conversion are the
AGOS code itself (third_party/AGOS), so the model matches training bit-for-bit. This module only
adapts real-robot observations to the AGOS rollout contract (policy/rollout_obs.py, To = 1):

* depth  [1, 1, 1, H, W]: wrist depth in Isaac Gym convention (negative metres along the view axis,
  no-hit = -inf -> "far"), then the checkpoint's DepthNormalizer (clip [-0.5, 0], fixed mean / std).
* canonical [1, 1, 2, 240, 320]: [plug, socket] raw canonical depth (NaN = empty) -> CanonicalNormalizer.
* force [1, 1, 3]: socket contact force in the world frame, clipped to +-clip_raw, then MinMax.

Actions: [10, 9] = (translation, rot6d) in the training frame (``eef`` for the sim2real checkpoints),
un-normalized, rotation-limited (cap / deadband, as in AGOS eval), and converted to world per executed
step with the *live* fingertip quaternion (``executed_action_to_world``).
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys

import numpy as np
import torch

AGOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "AGOS"))
TORCH_HUB_RESNET18 = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")


def _import_agos(agos_root: str):
    if agos_root not in sys.path:
        sys.path.insert(0, agos_root)
    from dataset.action_frames import executed_action_to_world, limit_action_rotations
    from diffusion_utils.normalization import NormalizationStats
    from policy.checkpoint import load_checkpoint
    from policy.multimodal_diffusion_policy import build_policy

    return load_checkpoint, build_policy, NormalizationStats, executed_action_to_world, limit_action_rotations


class AGOSPolicy:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        agos_root: str = AGOS_ROOT,
        resnet18_weights: str = TORCH_HUB_RESNET18,
        seed: int = 0,
        rot_cap_deg: float = 0.5,
        rot_deadband_deg: float = 0.3,
        force_clip_raw: float | None = None,
    ):
        (load_checkpoint, build_policy, NormalizationStats,
         self._executed_action_to_world, self._limit_action_rotations) = _import_agos(agos_root)
        payload = load_checkpoint(ckpt_path, map_location="cpu")  # schema + stats checksum checks
        self.device = torch.device(device)
        self.model_config = copy.deepcopy(payload["model_config"])
        self.data_shapes = payload["data_shapes"]
        data_cfg = payload["resolved_config"]["data"]
        self.modalities = list(self.model_config["modalities"])
        self.action_frame = data_cfg["action"]["frame"]
        self.force_clip_raw = float(force_clip_raw if force_clip_raw is not None
                                    else data_cfg.get("force", {}).get("clip_raw") or 0.0)
        self.depth_hw = tuple(self.data_shapes["depth"][-2:]) if "depth" in self.data_shapes else None
        self.canonical_hw = tuple(self.data_shapes["canonical"][-2:]) if "canonical" in self.data_shapes else None

        # ImageNet ResNet-18 init, as in training (the checkpoint stores a repo-relative path that only
        # exists on the training machine; torchvision's IMAGENET1K_V1 file is the same weights).
        for enc in self.model_config.get("encoders", {}).values():
            path = enc.get("pretrained_weights")
            if path and not os.path.isfile(path) and not os.path.isfile(os.path.join(agos_root, path)):
                if not os.path.isfile(resnet18_weights):
                    raise FileNotFoundError(f"ResNet-18 ImageNet weights not found: {path} / {resnet18_weights}")
                enc["pretrained_weights"] = resnet18_weights
        self.policy = build_policy(self.model_config, self.data_shapes)
        use_ema = bool(payload["resolved_config"]["training"].get("ema", {}).get("enabled", False))
        self.policy.load_state_dict(payload["ema_state" if use_ema else "model_state"], strict=True)
        self.policy.to(self.device).eval()
        self.stats = NormalizationStats.from_dict(payload["normalization_stats"])
        self.generator = torch.Generator(self.device).manual_seed(int(seed) + 900)  # AGOS eval seed offset
        self.rot_cap_deg, self.rot_deadband_deg = float(rot_cap_deg), float(rot_deadband_deg)
        with open(resnet18_weights, "rb") as f:
            resnet_sha = hashlib.sha256(f.read()).hexdigest()[:12]
        print(f"[agos] Loaded {self.model_config['name']} ({'EMA' if use_ema else 'raw'} weights, step "
              f"{payload['global_step']}) from {ckpt_path}; frame={self.action_frame}, modalities={self.modalities}, "
              f"depth {self.depth_hw}, canonical {self.canonical_hw}; ResNet-18 init sha256 {resnet_sha}")

    # ------------------------------------------------------------------ observations
    def depth_obs(self, depth_m: np.ndarray) -> torch.Tensor:
        """Real wrist depth (positive metres, 0/NaN = invalid) at the checkpoint resolution."""
        d = np.asarray(depth_m, dtype=np.float32)
        if self.depth_hw is not None and d.shape != self.depth_hw:
            raise ValueError(f"wrist depth {d.shape} != checkpoint depth {self.depth_hw}")
        valid = np.isfinite(d) & (d > 0)
        sim_depth = np.where(valid, -d, -np.inf).astype(np.float32)  # Isaac Gym: negative, no-hit = -inf
        normed = self.stats["depth"](sim_depth[None, None]).astype(np.float32)  # [N=1, 1, H, W]
        return torch.from_numpy(np.ascontiguousarray(normed)).to(self.device)[:, None]  # [1, To=1, 1, H, W]

    def canonical_obs(self, plug_depth: np.ndarray, socket_depth: np.ndarray) -> torch.Tensor:
        p = np.asarray(plug_depth, dtype=np.float32)
        s = np.asarray(socket_depth, dtype=np.float32)
        if self.canonical_hw is not None and (p.shape != self.canonical_hw or s.shape != self.canonical_hw):
            raise ValueError(f"canonical {p.shape}/{s.shape} != checkpoint {self.canonical_hw}")
        c = self.stats["canonical"](np.stack([p, s], axis=0)[None]).astype(np.float32)  # [1, 2, H, W]
        return torch.from_numpy(np.ascontiguousarray(c)).to(self.device)[:, None]

    def force_obs(self, socket_force_world: np.ndarray) -> torch.Tensor:
        f = np.asarray(socket_force_world, dtype=np.float32).reshape(1, 3)
        if self.force_clip_raw > 0:
            f = np.clip(f, -self.force_clip_raw, self.force_clip_raw)
        f = self.stats["force"](f).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(f)).to(self.device)[:, None]

    def build_obs(self, depth_m=None, plug_depth=None, socket_depth=None, socket_force_world=None) -> dict:
        obs = {}
        if "depth" in self.modalities:
            obs["depth"] = self.depth_obs(depth_m)
        if "canonical" in self.modalities:
            obs["canonical"] = self.canonical_obs(plug_depth, socket_depth)
        if "force" in self.modalities:
            obs["force"] = self.force_obs(socket_force_world)
        return obs

    # ------------------------------------------------------------------ actions
    @torch.no_grad()
    def predict(self, obs: dict) -> torch.Tensor:
        """[T, 9] actions in the training frame: un-normalized and rotation-limited (AGOS eval)."""
        pred_norm = self.policy.predict_action(obs, generator=self.generator)
        pred = self.stats["action"].inverse(pred_norm)
        if self.rot_cap_deg > 0:
            pred = self._limit_action_rotations(pred, self.rot_cap_deg, self.rot_deadband_deg)
        return pred[0].float().cpu()

    def action_to_world(self, action: torch.Tensor, fingertip_quat_xyzw: np.ndarray) -> np.ndarray:
        """One executed step -> world (dpos, rot6d) using the live fingertip quaternion (xyzw)."""
        a = torch.as_tensor(action, dtype=torch.float64)[None]
        q = torch.as_tensor(np.asarray(fingertip_quat_xyzw, dtype=np.float64))[None]
        return self._executed_action_to_world(a, q, self.action_frame)[0].numpy()
