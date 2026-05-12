"""
AMPLIFY policy inference server.
Run in the 'amplify' conda env BEFORE running control_robot.py in the lerobot env.

Usage:
    conda activate amplify
    cd /home/haotian/lerobot

    # With bundled checkpoint:
    python lerobot/common/policies/polaris_policy/amplify/amplify_server.py \\
        --ckpt_path /path/to/bundled.pt \\
        --text_emb /path/to/text_emb.npy \\
        --port 5558

    # With separate checkpoints:
    python lerobot/common/policies/polaris_policy/amplify/amplify_server.py \\
        --mt_ckpt  /path/to/motion_tokenizer.pt \\
        --fd_ckpt  /path/to/forward_dynamics.pt \\
        --id_ckpt  /path/to/inverse_dynamics.pt \\
        --text_emb /path/to/text_emb.npy \\
        --port 5558

    # With a no-motion inverse-dynamics-only checkpoint (trained via
    # train_inverse_dynamics_no_motion.py):
    python lerobot/common/policies/polaris_policy/amplify/amplify_server.py \\
        --no_motion \\
        --ckpt_path /path/to/inverse_dynamics_no_motion.pt \\
        --port 5558

Request format (pickle):
    {"image": np.ndarray (v, H, W, 3) float32 [0,1],
     "proprio": np.ndarray (10,) float32 = pos(3)+rot6d(6)+gripper(1)}
    or {"reset": True}

Response format (pickle):
    {"action_chunk": np.ndarray (action_horizon, 10) float32}
    actions are pos(3)+rot6d(6)+gripper(1), denormalized to original EEF pose space.
    or {"error": str}
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np
import torch
import zmq
from einops import repeat

# Make the polaris AMPLIFY package importable
_AMPLIFY_ROOT = str(
    pathlib.Path(__file__).parent.parent.parent.parent.parent.parent /
    "polaris" / "src" / "polaris" / "policy" / "amplify"
)
_POLARIS_SRC = str(
    pathlib.Path(__file__).parent.parent.parent.parent.parent.parent /
    "polaris" / "src"
)
sys.path.insert(0, _AMPLIFY_ROOT)
sys.path.insert(0, _POLARIS_SRC)

from amplify import AMPLIFY
from amplify.utils.kp_utils.query_utils import grid_queries_nonsquare
from amplify.utils.vis_utils import vis_pred

import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.inverse_dynamics import InverseDynamics


class NoMotionPolicy(nn.Module):
    """Wraps a vision encoder + inverse dynamics model trained without motion
    tokens / forward dynamics. Exposes an `act(images, proprio, ...)` interface
    matching AMPLIFY.act so the request handler can stay unified."""

    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])

        self.cfg = cfg
        self.norm_stats = ckpt.get("norm_stats", None)
        if self.norm_stats is not None:
            self.norm_stats = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in self.norm_stats.items()}

        # Synthetic motion_tokenizer_cfg, mirroring train_inverse_dynamics_no_motion.build_motion_tokenizer_cfg
        motion_tokenizer_cfg = OmegaConf.create({
            "track_method": cfg.track_method,
            "cond_cameraviews": list(cfg.cond_cameraviews),
            "img_shape": list(cfg.img_shape),
            "true_horizon": cfg.true_horizon,
            "track_pred_horizon": cfg.track_pred_horizon,
            "hidden_dim": cfg.hidden_dim,
            "num_tracks": cfg.num_tracks,
            "interp_method": cfg.interp_method,
            "per_view": cfg.per_view,
            "keys_to_load": ["images"],
            "pick_mug_hdf5_path": OmegaConf.to_container(cfg.pick_mug_hdf5_path) if "pick_mug_hdf5_path" in cfg else None,
            "libero_path": None,
        })
        self.motion_tokenizer_cfg = motion_tokenizer_cfg
        self.num_views = len(motion_tokenizer_cfg.cond_cameraviews)

        # Vision encoder
        self.img_encoder = VisionEncoder(**cfg.vision_encoder).eval()
        cfg.num_img_tokens = self.img_encoder.seq_len
        cfg.img_embed_dim = self.img_encoder.embed_dim * self.num_views

        # Inverse dynamics
        self.inverse_dynamics = InverseDynamics(motion_tokenizer_cfg, cfg)
        self.inverse_dynamics.load_state_dict(ckpt["model"], strict=False)

        self.to(device)
        self._dev = device

    @property
    def device(self):
        return self._dev

    @torch.no_grad()
    def act(self, images: torch.Tensor, proprio: torch.Tensor = None, text_emb=None, ar_sampling=None, **_):
        b, v = images.shape[0], images.shape[1]
        # import pdb; pdb.set_trace()
        img = rearrange(images, "b v h w c -> (b v) h w c")
        img_tokens = self.img_encoder(img)
        img_tokens = rearrange(img_tokens, "(b v) t d -> b t (v d)", v=v)

        input_dict = {"img_tokens": img_tokens}

        if self.cfg.cond_on_proprio:
            assert proprio is not None, "proprio required (cfg.cond_on_proprio=True)"
            p = proprio.to(self.device)
            if self.norm_stats is not None and self.norm_stats.get("proprio_min") is not None:
                pmin = self.norm_stats["proprio_min"].to(self.device)
                pmax = self.norm_stats["proprio_max"].to(self.device)
                p = (p - pmin) / (pmax - pmin + 1e-8) * 2.0 - 1.0
            input_dict["proprioception"] = p.unsqueeze(1)

        if self.cfg.cond_on_text:
            assert text_emb is not None, "text_emb required (cfg.cond_on_text=True)"
            te = text_emb.to(self.device)
            if te.dim() == 2:
                te = te.unsqueeze(1)
            input_dict["text_tokens"] = te

        actions = self.inverse_dynamics.act(input_dict)

        if self.norm_stats is not None:
            amin = self.norm_stats["min"].to(actions.device)
            amax = self.norm_stats["max"].to(actions.device)
            actions = (actions + 1.0) / 2.0 * (amax - amin + 1e-8) + amin
        return actions


def load_policy(args):
    # Load on CPU first to avoid CUDA availability issues in the conda env,
    # then move to GPU after loading.
    cpu = torch.device("cpu")
    if args.no_motion:
        print(f"Loading no-motion inverse-dynamics policy from {args.ckpt_path}...")
        policy = NoMotionPolicy(args.ckpt_path, device=torch.device("cuda:0"))
        policy.eval()
        return policy
    if args.ckpt_path:
        print(f"Loading bundled AMPLIFY from {args.ckpt_path}...")
        policy = AMPLIFY.load(args.ckpt_path, device=cpu)
    else:
        print("Bundling AMPLIFY from separate checkpoints...")
        print(f"  MT: {args.mt_ckpt}")
        print(f"  FD: {args.fd_ckpt}")
        print(f"  ID: {args.id_ckpt}")
        policy, saved = AMPLIFY.bundle(
            motion_tokenizer_ckpt=args.mt_ckpt,
            forward_dynamics_ckpt=args.fd_ckpt,
            inverse_dynamics_ckpt=args.id_ckpt,
            save_to=args.bundle_save,
        )
        if saved:
            print(f"Bundled model saved to: {saved}")
    policy.eval()
    policy.to("cuda:0")
    return policy


def main():
    parser = argparse.ArgumentParser(description="AMPLIFY inference server")
    parser.add_argument("--amplify_root", type=str, default=None,
                        help="Override path to AMPLIFY repo root (default: auto-detected)")
    # Checkpoint options
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to bundled AMPLIFY checkpoint (.pt)")
    parser.add_argument("--mt_ckpt", type=str, default=None,
                        help="Path to motion tokenizer checkpoint")
    parser.add_argument("--fd_ckpt", type=str, default=None,
                        help="Path to forward dynamics checkpoint")
    parser.add_argument("--id_ckpt", type=str, default=None,
                        help="Path to inverse dynamics checkpoint")
    parser.add_argument("--bundle_save", type=str, default=None,
                        help="Save bundled checkpoint to this path (optional)")
    parser.add_argument("--no_motion", action="store_true",
                        help="Load a no-motion inverse-dynamics-only checkpoint "
                             "(trained via train_inverse_dynamics_no_motion.py). "
                             "In this mode --ckpt_path points to that checkpoint, "
                             "and --text_emb / --vis_tracks are ignored unless "
                             "the checkpoint was trained with cond_on_text=True.")
    # Task artifacts
    parser.add_argument("--text_emb", type=str, default=None,
                        help="Path to .npy text embedding (1, 512) or (512,). "
                             "Required for bundled mode; optional for --no_motion.")
    # Server
    parser.add_argument("--port", type=int, default=5558)
    # Track visualization
    parser.add_argument("--vis_tracks", action="store_true",
                        help="Predict and return track visualization each step")
    parser.add_argument("--num_tracks", type=int, default=400,
                        help="Target n_tracks for grid_queries_nonsquare (default: 400)")
    parser.add_argument("--orig_h", type=int, default=240,
                        help="Image height used during track preprocessing (default: 240)")
    parser.add_argument("--orig_w", type=int, default=426,
                        help="Image width used during track preprocessing (default: 426)")
    args = parser.parse_args()

    if args.amplify_root:
        sys.path.insert(0, args.amplify_root)

    if args.no_motion:
        if not args.ckpt_path:
            parser.error("--no_motion requires --ckpt_path pointing to the inverse-dynamics-only checkpoint")
    elif not args.ckpt_path and not (args.mt_ckpt and args.fd_ckpt and args.id_ckpt):
        parser.error("Provide either --ckpt_path or all of --mt_ckpt --fd_ckpt --id_ckpt")
    if not args.no_motion and args.text_emb is None:
        parser.error("--text_emb is required for bundled AMPLIFY mode")

    # Load policy
    policy = load_policy(args)

    # In no_motion mode, only load text_emb if the trained model uses text conditioning
    text_emb_t = None
    needs_text_emb = (not args.no_motion) or bool(getattr(policy, "cfg", None) and policy.cfg.cond_on_text)
    if needs_text_emb:
        if args.text_emb is None:
            parser.error("--text_emb is required because the policy was trained with cond_on_text=True")
        text_emb = np.load(args.text_emb)
        if text_emb.ndim == 1:
            text_emb = text_emb[None]                                    # (1, 512)
        text_emb_t = torch.from_numpy(text_emb).float().to("cuda:0")    # (1, 512)
        print(f"Text embedding loaded: shape={text_emb_t.shape}")
    else:
        print("No-motion mode with cond_on_text=False: skipping text embedding.")

    # Pre-compute grid queries for track visualization
    init_queries = None
    if args.vis_tracks and args.no_motion:
        print("--vis_tracks ignored in --no_motion mode (no motion tokenizer to decode tracks).")
        args.vis_tracks = False
    if args.vis_tracks:
        init_queries = grid_queries_nonsquare(
            views=1, n_tracks=args.num_tracks, device=torch.device("cuda:0"),
            image_height=args.orig_h, image_width=args.orig_w,
        ).standard()
        actual_n = init_queries.shape[1]
        print(f"Track visualization: {actual_n} grid tracks ({args.orig_h}×{args.orig_w})")

    # ZMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"AMPLIFY server listening on port {args.port}")

    while True:
        raw = socket.recv()
        try:
            request = pickle.loads(raw)
        except Exception as e:
            print(f"[Server] Deserialize error: {e}")
            socket.send(pickle.dumps({"error": str(e)}))
            continue

        if request.get("reset"):
            socket.send(pickle.dumps({"status": "ok"}))
            continue

        try:
            # image:   (v, H, W, 3) float32 [0,1]
            # proprio: (10,) float32 = pos(3)+rot6d(6)+gripper(1)
            image_np   = request["image"]    # (v, H, W, 3)
            proprio_np = request["proprio"]  # (10,)

            image_t   = torch.from_numpy(image_np).float().unsqueeze(0).to("cuda:0")    # (1, v, H, W, 3)
            proprio_t = torch.from_numpy(proprio_np).float().unsqueeze(0).to("cuda:0")  # (1, 10)

            with torch.no_grad():
                # the predicted action is denormalized inside the act() function if ckpt has the norm stats
                actions = policy.act(
                    images=image_t,
                    proprio=proprio_t,
                    text_emb=text_emb_t,
                    ar_sampling="argmax",
                )  # (1, action_horizon, 10)

            # action_chunk is in denormalized version
            action_chunk = actions.squeeze(0).cpu().numpy()   # (action_horizon, 10)
            response = {"action_chunk": action_chunk}

            if args.vis_tracks and init_queries is not None:
                num_views = image_t.shape[1]
                traj_queries = repeat(init_queries, "1 n d -> b v 1 n d", b=1, v=num_views)
                with torch.no_grad():
                    pred_traj = policy.predict_traj(
                        images=image_t,
                        init_queries=traj_queries,
                        text_emb=text_emb_t,
                        ar_sampling="argmax",
                    )  # (1, v, t+1, n, 2)
                vis_frame = vis_pred(image_t, pred_traj)  # (1, H, v*W, 3) uint8
                response["vis_frame"] = vis_frame[0].cpu().numpy()

        except Exception as e:
            import traceback
            print(f"[Server] Inference error: {e}\n{traceback.format_exc()}")
            socket.send(pickle.dumps({"error": str(e)}))
            continue

        socket.send(pickle.dumps(response))


if __name__ == "__main__":
    main()
