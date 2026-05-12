"""
Polaris inference server for chi-style diffusion policies from polaris/src/polaris/policy.
Run this in the robodiff conda env BEFORE running control_robot.py in the lerobot env.

Usage:
    conda activate robodiff
    cd /home/haotian/lerobot
    python lerobot/common/policies/polaris_server.py \\
        --ckpt_path <path/to/checkpoint.ckpt> \\
        --open_loop_horizon 6 \\
        --port 5557

The server:
  1. Loads a polaris DiffusionUnetHybridImagePolicy checkpoint.
  2. Buffers n_obs_steps observations per key.
  3. Runs inference when the action queue is empty (or after open_loop_horizon steps).
  4. Returns one action per request.

The client (PolarisPolicy in lerobot) sends per-step observations as numpy dicts
and receives one action numpy array per step.

Request format (pickle):
    {"reset": True}                              — reset buffers
    {"obs": {key: np.ndarray}, "task": str}     — observation step

The obs values have batch dim stripped: e.g. images are (3, H, W) float32 [0,1],
state is (D,) float32.

Response format (pickle):
    {"status": "ok"}                            — after reset
    {"action": np.ndarray, "action_eef": np.ndarray}   — action step
    {"error": str}                               — on failure
"""

import argparse
import pathlib
import pickle
import sys
import traceback
from collections import deque

import numpy as np
import torch
import zmq
import dill
from unittest.mock import MagicMock
import types


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


# Stub wandb before importing the workspace (avoids tempfile creation issues)
_wandb_mock = MagicMock()
_wandb_mock.__spec__ = types.ModuleType("wandb")
sys.modules["wandb"] = _wandb_mock

# Make the polaris diffusion_policy package importable
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.parent /
               "polaris" / "src" / "polaris" / "policy" / "diffusion_policy")
sys.path.insert(0, ROOT_DIR)


def load_policy(ckpt_path: str):
    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    workspace_type = cfg.get("_target_", "")
    if "DiffusionTransformerHybrid" in workspace_type or "diffusion_transformer_hybrid" in str(cfg):
        from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (
            TrainDiffusionTransformerHybridWorkspace as WorkspaceCls,
        )
    else:
        from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
            TrainDiffusionUnetHybridWorkspace as WorkspaceCls,
        )

    workspace = WorkspaceCls(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.get("use_ema", True) else workspace.model
    policy.eval()
    policy.to("cuda")
    return policy, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--open_loop_horizon", type=int, default=None,
                        help="Steps to execute per inference. Defaults to policy n_action_steps.")
    parser.add_argument("--port", type=int, default=5557)
    args = parser.parse_args()

    print(f"Loading policy from {args.ckpt_path} ...")
    policy, cfg = load_policy(args.ckpt_path)
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = policy.n_action_steps
    open_loop_horizon = args.open_loop_horizon if args.open_loop_horizon is not None else n_action_steps
    print(f"Policy loaded. n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}, "
          f"open_loop_horizon={open_loop_horizon}")

    obs_bufs: dict[str, deque] = {}
    action_buf: deque = deque()
    steps_from_last_inference = 0

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Polaris server listening on port {args.port}")

    while True:
        raw = socket.recv()
        try:
            request = _deserialize(raw)
        except Exception as e:
            socket.send(_serialize({"error": f"Deserialization failed: {e}"}))
            continue

        if request.get("reset"):
            obs_bufs.clear()
            action_buf.clear()
            steps_from_last_inference = 0
            socket.send(_serialize({"status": "ok"}))
            continue

        obs_np: dict = request["obs"]

        if not obs_bufs:
            for key, val in obs_np.items():
                obs_bufs[key] = deque(maxlen=n_obs_steps)

        for key, val in obs_np.items():
            obs_bufs[key].append(val)

        for key in obs_bufs:
            while len(obs_bufs[key]) < n_obs_steps:
                obs_bufs[key].appendleft(obs_bufs[key][0])

        need_inference = (len(action_buf) == 0 or steps_from_last_inference >= open_loop_horizon)
        if need_inference:
            obs_dict = {}
            for key in obs_bufs:
                stacked = np.stack(list(obs_bufs[key]), axis=0)
                obs_dict[key] = torch.from_numpy(stacked).float().unsqueeze(0).to("cuda")

            with torch.no_grad():
                result = policy.predict_action(obs_dict)

            action_chunk = result["action"].squeeze(0).cpu().numpy()

            action_buf.clear()
            for i in range(min(open_loop_horizon, len(action_chunk))):
                action_buf.append(action_chunk[i])
            steps_from_last_inference = 0

        action = action_buf.popleft()
        steps_from_last_inference += 1

        socket.send(_serialize({"action": action, "action_eef": action}))


if __name__ == "__main__":
    main()
