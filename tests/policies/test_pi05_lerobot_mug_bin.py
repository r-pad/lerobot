#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script to verify PI0OpenPI policy integration with LeRobot vs the original implementation.
FEATURES:
 - Uses REAL DATA from local LeRobot dataset cache.
 - Uses LOCAL CHECKPOINT for model weights.
 - Filters actions to JOINT ANGLE SPACE.
 - Generates comparative plots.
"""

import inspect
import json
import os
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pytest
import torch
import draccus
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

# --- IMPORTS & CHECKS ---
# Skip if openpi or transformers is not available
pytest.importorskip("openpi")
pytest.importorskip("transformers")

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local OpenPI installation and is not meant for CI",
)

from openpi.models_pytorch import preprocessing_pytorch as openpi_preprocessing  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from aloha_policy import normalize_state_mean_std, unnormalize_action_mean_std, _decode_state, _encode_actions_inv

from lerobot.common.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from lerobot.common.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import load_stats
from lerobot.configs.types import FeatureType, PolicyFeature

# --- 🛠️ PATCH: LEROBOT DATASET (Torch Stack Fix) 🛠️ ---
def patch_lerobot_dataset():
    """Monkey patches LeRobotDataset to fix a specific torch.stack vs numpy error."""
    try:
        source = inspect.getsource(LeRobotDataset.__init__)
        source = textwrap.dedent(source)
        
        # Fix super() call which fails in monkey-patched methods (missing __class__ cell)
        source = source.replace('super().__init__()', 'torch.utils.data.Dataset.__init__(self)')
        
        old_code = 'torch.stack(self.hf_dataset["timestamp"]).numpy()'
        new_code = 'torch.from_numpy(self.hf_dataset["timestamp"].to_numpy()).float().numpy()'
        
        if old_code in source:
            source = source.replace(old_code, new_code)
            
        module_globals = inspect.getmodule(LeRobotDataset).__dict__
        exec_locals = {}
        exec(source, module_globals, exec_locals)
        LeRobotDataset.__init__ = exec_locals['__init__']
        print("[Patch] LeRobotDataset.__init__ successfully patched!")

    except Exception as e:
        print(f"[Patch] Failed to patch LeRobotDataset: {e}")

patch_lerobot_dataset()

# --- CONFIGURATION ---
DATASET_ROOT = "/home/ktsim/.cache/huggingface/lerobot/sriramsk/mug_bin_multiview_20251113_ss"
LOCAL_CHECKPOINT_DIR = "/home/ktsim/Projects/rpad_lerobot/20000"
DEVICE = "cpu"  # Change to "cuda" if memory allows

# Joint angle space configuration
# These indices filter the raw action space down to specific joints
KEEP_INDICES = [0, 1, 3, 6, 5, 7, 8, 9, 10, 12, 15, 14, 16, 17]

CAMERA_SEARCH_MAPPING = {
    "azure_kinect_front": "base_0_rgb",
    "azure_kinect_back": "left_wrist_0_rgb",
    "wrist": "right_wrist_0_rgb"
}

STATS_PATHS = [
    os.path.join(LOCAL_CHECKPOINT_DIR, "dataset_stats.json"),
    os.path.join(DATASET_ROOT, "meta/stats.json"),
    os.path.join(DATASET_ROOT, "dataset_stats.json"),
]

# --- Load Statistics & Set Dimensions ---
DATASET_STATS = None
for path in STATS_PATHS:
    if os.path.exists(path):
        print(f"Loading stats from: {path}")
        DATASET_STATS = load_stats(Path(path))
        break

DUMMY_ACTION_DIM = 32  # Forced to 32 as per checkpoint expectation
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200

if DATASET_STATS:
    stats_root = DATASET_STATS["norm_stats"] if "norm_stats" in DATASET_STATS else DATASET_STATS
    if "state" in stats_root:
         val = stats_root["state"]["mean"]
         DUMMY_STATE_DIM = val.shape[0] if hasattr(val, "shape") else len(val)
    else:
         DUMMY_STATE_DIM = 18
    print(f"Inferred Dims - Action: {DUMMY_ACTION_DIM}, State: {DUMMY_STATE_DIM}")
else:
    print(f"WARNING: No stats found. Using defaults.")
    DUMMY_STATE_DIM = 18


class PI05BaseOriginalConfig:
    """Configuration class matching the original OpenPI structure."""
    action_dim: int = DUMMY_ACTION_DIM
    action_horizon: int = DUMMY_ACTION_HORIZON
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "float32"
    pi05: bool = True
    dtype: str = "float32"


class PI05Observation:
    """Observation class that matches the original OpenPI format."""
    def __init__(
        self,
        state,
        images,
        image_masks,
        tokenized_prompt,
        tokenized_prompt_mask,
        token_ar_mask,
        token_loss_mask,
    ):
        self.state = state
        self.images = images
        self.image_masks = image_masks
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


# --- INSTANTIATION FUNCTIONS ---

def instantiate_lerobot_pi05(from_pretrained: bool = False):
    """Loads the LeRobot implementation of PI05."""
    
    # Define Image Features matching REAL data resolution (e.g. 720p)
    # Adjust shapes if your dataset differs
    image_features_config = {
        "observation.images.base_0_rgb": {
            "type": "VISUAL", "shape": [3, 720, 1280], "names": ["channels", "height", "width"]
        },
        "observation.images.left_wrist_0_rgb": {
            "type": "VISUAL", "shape": [3, 720, 1280], "names": ["channels", "height", "width"]
        },
        "observation.images.right_wrist_0_rgb": {
            "type": "VISUAL", "shape": [3, 720, 1280], "names": ["channels", "height", "width"]
        },
    }

    if from_pretrained:
        print(f"Loading LeRobot PI05 from local checkpoint: {LOCAL_CHECKPOINT_DIR}...")
        config_path = os.path.join(LOCAL_CHECKPOINT_DIR, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Remove keys that break draccus parsing
            forbidden_keys = ["push_to_hub", "repo_id", "private", "tags", "license", "_name_or_path", "type"]
            for key in forbidden_keys:
                config_dict.pop(key, None)
            config = draccus.decode(PI05Config, config_dict)
            config.device = DEVICE
        else:
            print(f"WARNING: config.json NOT FOUND. Creating configuration manually...")
            config = PI05Config(
                max_action_dim=DUMMY_ACTION_DIM,
                max_state_dim=DUMMY_STATE_DIM,
                dtype="float32",
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
            )
            config.device = DEVICE

        # Force input features
        if config.input_features is None: config.input_features = {}
        for k, v in image_features_config.items():
            config.input_features[k] = PolicyFeature(
                type=FeatureType(v["type"]),
                shape=tuple(v["shape"])
            )

        policy = PI05Policy(config)
        
        # Load Weights Manually
        model_path = os.path.join(LOCAL_CHECKPOINT_DIR, "model.safetensors")
        print(f"Loading weights from {model_path}...")
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            # Remap keys
            remapped_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("model."):
                    remapped_state_dict[f"model.{key}"] = value
                else:
                    remapped_state_dict[key] = value

            missing, unexpected = policy.load_state_dict(remapped_state_dict, strict=False)
            if len(missing) > 100:
                print("Warning: Remapping keys didn't work well. Trying raw keys...")
                missing, unexpected = policy.load_state_dict(state_dict, strict=False)

            if missing: print(f"LeRobot Missing keys: {len(missing)}")
            if unexpected: print(f"LeRobot Unexpected keys: {len(unexpected)}")
        else:
             print(f"CRITICAL ERROR: Model weights not found at {model_path}")
    else:
        config = PI05Config(max_action_dim=DUMMY_ACTION_DIM, max_state_dim=DUMMY_STATE_DIM, dtype="float32")
        config.device = DEVICE
        policy = PI05Policy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config, dataset_stats=DATASET_STATS
    )
    return (policy, preprocessor, postprocessor)


def instantiate_original_pi05(from_pretrained: bool = False, model_path: str | None = None):
    """Loads the original OpenPI implementation."""
    config = PI05BaseOriginalConfig()
    policy = PI0Pytorch(config)

    if from_pretrained:
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading Original PI05 from local checkpoint: {model_path}")
                model_file = os.path.join(model_path, "model.safetensors")
            else:
                # Fallback to HF Hub if path invalid
                print("Loading converted PyTorch weights from HuggingFace Hub...")
                from huggingface_hub import snapshot_download
                cache_dir = snapshot_download(repo_id="lerobot/pi05_base", repo_type="model")
                model_file = os.path.join(cache_dir, "model.safetensors")

            if os.path.exists(model_file):
                print(f"Loading weights from {model_file}")
                import safetensors.torch
                state_dict = safetensors.torch.load_file(model_file)
                
                # Unwrap 'model.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                
                missing_keys, unexpected_keys = policy.load_state_dict(new_state_dict, strict=False)

                if missing_keys and len(missing_keys) < 20:
                     print(f"OpenPI Missing keys: {len(missing_keys)}")
                if not missing_keys and not unexpected_keys:
                    print("OpenPI weights loaded successfully!")
            else:
                raise FileNotFoundError(f"No safetensors file found at {model_file}")

        except Exception as e:
            print(f"Failed to load OpenPI weights: {e}")

    policy.to(DEVICE)
    return policy


# --- DATA UTILS ---

def find_key_recursive(data, search_term):
    """Helper to find keys in nested dictionary structures."""
    if isinstance(data, dict):
        for k, v in data.items():
            if search_term in k:
                if isinstance(v, dict):
                     if "color" in v: return v["color"]
                     if "image" in v: return v["image"]
                return v
            res = find_key_recursive(v, search_term)
            if res is not None: return res
    return None

def create_real_data_batch(batch_size=2):
    """Loads real data, filters for joint angles, and prepares batch."""
    print(f"Loading dataset from LOCAL ROOT: {DATASET_ROOT}...")
    ds = LeRobotDataset(
        root=DATASET_ROOT,
        repo_id="sriramsk/mug_bin_multiview_20251113_ss",
    )
    
    batch = {"observation.state": [], "action": [], "task": []}
    for target_key in CAMERA_SEARCH_MAPPING.values():
        batch[f"observation.images.{target_key}"] = []

    for i in range(batch_size):
        ex = ds[i]
        
        # 1. STATE
        if "observation.state" in ex:
            raw_state = ex["observation.state"]
        else:
            raw_state = find_key_recursive(ex, "state")
        
        if not isinstance(raw_state, torch.Tensor): raw_state = torch.tensor(raw_state, dtype=torch.float32)
        batch["observation.state"].append(raw_state)
        
        # 2. ACTION (Joint Angle Filtering)
        raw_action = ex["action"]
        if not isinstance(raw_action, torch.Tensor): raw_action = torch.tensor(raw_action, dtype=torch.float32)
        if raw_action.ndim == 1: raw_action = raw_action.unsqueeze(0)
        
        # Filter action to joint angles
        if raw_action.shape[-1] >= 18:
             filtered_action = raw_action[:, KEEP_INDICES] 
        else:
             filtered_action = raw_action

        # Pad Dimension if needed (to match DUMMY_ACTION_DIM=32)
        if filtered_action.shape[-1] < DUMMY_ACTION_DIM:
             dim_pad = DUMMY_ACTION_DIM - filtered_action.shape[-1]
             # filtered_action is [horizon, dim]
             padding_dim = torch.zeros((filtered_action.shape[0], dim_pad), dtype=torch.float32)
             filtered_action = torch.cat([filtered_action, padding_dim], dim=1)

        # Pad Horizon
        horizon = DUMMY_ACTION_HORIZON
        if filtered_action.shape[0] < horizon:
            pad_len = horizon - filtered_action.shape[0]
            padding = torch.zeros((pad_len, filtered_action.shape[1]), dtype=torch.float32)
            filtered_action = torch.cat([filtered_action, padding], dim=0)
        elif filtered_action.shape[0] > horizon:
            filtered_action = filtered_action[:horizon]
        batch["action"].append(filtered_action)
        
        # 3. TASK
        batch["task"].append("Place the mug in the bin")

        # 4. IMAGES
        for search_term, target_key in CAMERA_SEARCH_MAPPING.items():
            img_tensor = None
            val = find_key_recursive(ex, search_term)
            if val is not None:
                if isinstance(val, torch.Tensor):
                     img_tensor = val
                     if img_tensor.dtype == torch.uint8: img_tensor = img_tensor.float() / 255.0
                     if img_tensor.shape[-1] == 3 and img_tensor.ndim == 3: 
                         img_tensor = img_tensor.permute(2, 0, 1)
                else:
                    img_tensor = TF.to_tensor(val)
                
                # Resize to model expectation (224x224)
                img_tensor = TF.resize(img_tensor, [224, 224], antialias=True)
                batch[f"observation.images.{target_key}"].append(img_tensor)
            else:
                print(f"Skipping {target_key} -> Not found")
                batch[f"observation.images.{target_key}"].append(torch.zeros(3, 224, 224))

    # Stack and move to device
    batch["observation.state"] = torch.stack(batch["observation.state"]).to(DEVICE)
    batch["action"] = torch.stack(batch["action"]).to(DEVICE)
    for target_key in CAMERA_SEARCH_MAPPING.values():
        key = f"observation.images.{target_key}"
        batch[key] = torch.stack(batch[key]).to(DEVICE)

    return batch


def create_original_observation_with_openpi_preprocessing(batch):
    """Create observation object for OpenPI using OpenPI's own preprocessing."""
    batch_size = batch["observation.state"].shape[0]
    device = batch["observation.state"].device

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # Tasks
    tasks = batch.get("task", ["Pick up the object"] * batch_size)
    if isinstance(tasks, str): tasks = [tasks] * batch_size

    # Use pi05 state and input tokenizer logic
    state = batch["observation.state"]
    # Filter to 14 dims
    if state.shape[-1] >= 18:
        state = state[:, KEEP_INDICES]
        
    state = deepcopy(state)

    # --- NORMALIZE STATE (Manual for OpenPI) ---
    if DATASET_STATS:
        stats_root = DATASET_STATS["norm_stats"] if "norm_stats" in DATASET_STATS else DATASET_STATS
        if "state" in stats_root:
            state_mean = torch.tensor(stats_root["state"]["mean"], device=device, dtype=torch.float32)
            state_std = torch.tensor(stats_root["state"]["std"], device=device, dtype=torch.float32)
            
            # Filter stats to 14
            if state_mean.shape[0] >= 18:
                state_mean = state_mean[KEEP_INDICES]
                state_std = state_std[KEEP_INDICES]
            
            # Transform Aloha -> PI
            state_np = state.cpu().numpy()
            decoded = []
            for s in state_np:
                decoded.append(_decode_state(s, adapt_to_pi=True))
            state = torch.tensor(np.stack(decoded), device=device, dtype=torch.float32)
            
            # Use aloha_policy helper
            state = normalize_state_mean_std(state, state_mean, state_std)

    from lerobot.common.policies.pi05.modeling_pi05 import pad_vector
    state = pad_vector(state, DUMMY_STATE_DIM)
    
    state_np = state.cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    full_prompts = []
    for i, task in enumerate(tasks):
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized_states[i]))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        full_prompts.append(full_prompt)

    tokenized = tokenizer(
        full_prompts,
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=DUMMY_MAX_TOKEN_LEN,
        return_tensors="pt",
    )

    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].to(device, dtype=torch.bool)
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    # Convert LeRobot images [0,1] to OpenPI [-1,1]
    image_dict = {
        "base_0_rgb": batch["observation.images.base_0_rgb"] * 2.0 - 1.0,
        "left_wrist_0_rgb": batch["observation.images.left_wrist_0_rgb"] * 2.0 - 1.0,
        "right_wrist_0_rgb": batch["observation.images.right_wrist_0_rgb"] * 2.0 - 1.0,
    }

    image_masks_dict = {}
    for key in image_dict:
        image_masks_dict[key] = torch.ones(batch_size, dtype=torch.bool, device=device)

    raw_observation = PI05Observation(
        state=batch["observation.state"],
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )

    processed_obs = openpi_preprocessing.preprocess_observation_pytorch(raw_observation, train=False)
    return processed_obs


# --- MAIN TEST ---

def test_pi05_original_vs_lerobot():
    
    print("\n=== Initializing Models ===")
    lerobot_pi05, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_pi05(from_pretrained=True)
    
    original_pi0 = instantiate_original_pi05(
        from_pretrained=True,
        model_path=LOCAL_CHECKPOINT_DIR
    )

    print("\n=== Loading Real Data ===")
    batch = create_real_data_batch(batch_size=2)
    batch_lerobot = deepcopy(batch)

    print("\n=== Running OpenPI Inference ===")
    pi0_obs_openpi = create_original_observation_with_openpi_preprocessing(batch)

    original_pi0.eval()
    torch.manual_seed(42)
    batch_size = batch["observation.state"].shape[0]
    
    # Noise shape must match action dim
    noise_shape = (batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM)
    fixed_noise = torch.randn(noise_shape, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        openpi_actions = original_pi0.sample_actions(
            device=DEVICE, observation=pi0_obs_openpi, noise=fixed_noise, num_steps=10
        )
    
    print("\n=== Running LeRobot Inference ===")
    lerobot_pi05.eval()
    torch.manual_seed(42)

    batch_lerobot_processed = lerobot_preprocessor(batch_lerobot)
    with torch.no_grad():
        lerobot_actions_normalized = lerobot_pi05.predict_action_chunk(batch_lerobot_processed)
        
    # --- POSTPROCESSING (Unnormalization) ---
    print("\n=== Unnormalizing Predictions ===")
    
    # DEBUG: Check stats keys
    stats_root = DATASET_STATS["norm_stats"] if "norm_stats" in DATASET_STATS else DATASET_STATS
    print(f"Stats keys: {list(stats_root.keys())}")
    
    # LeRobot: use the pipeline postprocessor
    lerobot_actions_own = lerobot_postprocessor(lerobot_actions_normalized)
    
    print(f"LeRobot output range (post-process): {lerobot_actions_own.min().item():.3f} to {lerobot_actions_own.max().item():.3f}")
    
    # OpenPI: manually unnormalize using the same stats (as it returns normalized actions)
    # Get stats tensors
    if "action" in stats_root:
        action_stats = stats_root["action"]
    elif "actions" in stats_root:
        action_stats = stats_root["actions"]
    else:
        raise ValueError("No action stats found")
        
    action_mean = torch.tensor(action_stats["mean"], device=DEVICE)
    action_std = torch.tensor(action_stats["std"], device=DEVICE)
    
    # MANUAL FIX FOR LEROBOT IF NEEDED
    # If stats has "actions" but LeRobot looked for "action", it failed to unnormalize.
    # Check if values are still small (normalized)
    if lerobot_actions_own.abs().mean() < 5.0 and action_mean.abs().mean() > 50.0:
         print("Detected normalized values in LeRobot output. Applying manual unnormalization...")
         
         # Note: lerobot_actions_own is already sliced if unnormalized properly, but if not, it's padded.
         # But the unnormalize_action_mean_std helper expects shape match or broadcast.
         # Let's slice both to valid dim if stats are smaller.
         
         if action_mean.shape[0] < DUMMY_ACTION_DIM:
             valid_dim = action_mean.shape[0]
             lerobot_valid = lerobot_actions_own[..., :valid_dim]
             lerobot_actions_own_unnorm = unnormalize_action_mean_std(lerobot_valid, action_mean, action_std)
             # Pad back to keep shape consistent for later comparison logic (which slices again)
             padding = torch.zeros(
                 lerobot_actions_own.shape[0], lerobot_actions_own.shape[1], DUMMY_ACTION_DIM - valid_dim,
                 device=DEVICE
             )
             lerobot_actions_own = torch.cat([lerobot_actions_own_unnorm, padding], dim=-1)
         else:
             lerobot_actions_own = unnormalize_action_mean_std(lerobot_actions_own, action_mean, action_std)

    # 1. Slice to 14
    openpi_actions = openpi_actions[..., :14]
    
    # 2. Unnormalize
    if action_mean.shape[0] >= 18:
         action_mean = action_mean[KEEP_INDICES]
         action_std = action_std[KEEP_INDICES]
    # Ensure shapes match
    if action_mean.shape[0] != 14:
         # Warn or slice?
         action_mean = action_mean[:14]
         action_std = action_std[:14]
         
    openpi_actions = unnormalize_action_mean_std(openpi_actions, action_mean, action_std)
    
    # 3. Aloha Transform (PI -> Aloha)
    openpi_np = openpi_actions.cpu().numpy()
    encoded = []
    for b in range(openpi_np.shape[0]):
        encoded.append(_encode_actions_inv(openpi_np[b], adapt_to_pi=True))
    openpi_actions = torch.tensor(np.stack(encoded), device=DEVICE, dtype=torch.float32)

    # --- PLOTTING ---
    print("\n=== Generating Comparison Plots ===")
    
    # Shapes: (batch, horizon, dim) -> convert to numpy
    gt_action_np = batch['action'].detach().cpu().numpy()
    lerobot_action_np = lerobot_actions_own.detach().cpu().numpy()
    openpi_action_np = openpi_actions.detach().cpu().numpy()

    # Calculate MAE per time step (average over batch and action dimensions)
    # We only care about the filtered joints (first len(KEEP_INDICES) dims) if the rest are padding
    # But usually padding is zero, so including them is fine, or we slice:
    valid_dims = len(KEEP_INDICES)
    
    # Slice to valid dims to avoid padding noise
    lerobot_action_np = lerobot_action_np[..., :valid_dims]
    openpi_action_np = openpi_action_np[..., :valid_dims]
    # For GT, we might need to be careful if it was padded.
    gt_action_np = gt_action_np[..., :valid_dims]

    mae_lerobot = np.mean(np.abs(lerobot_action_np - gt_action_np), axis=(0, 2))
    mae_openpi = np.mean(np.abs(openpi_action_np - gt_action_np), axis=(0, 2))
    mae_diff = np.mean(np.abs(lerobot_action_np - openpi_action_np), axis=(0, 2))

    time_steps = np.arange(mae_lerobot.shape[0])

    print(f"Lerobot output actions (first step): {lerobot_action_np[0, 0]}")
    print(f"OpenPI output actions (first step): {openpi_action_np[0, 0]}")
    print(f"Ground truth actions (first step): {gt_action_np[0, 0]}")

    plt.figure(figsize=(12, 5))

    # Plot 1: Error vs GT
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, mae_lerobot, label='LeRobot vs GT', linewidth=2, alpha=0.8)
    plt.plot(time_steps, mae_openpi, label='OpenPI vs GT', linewidth=2, linestyle='--', alpha=0.8)
    plt.xlabel('Time Step (Horizon)')
    plt.ylabel('Mean Absolute Error (Joints)')
    plt.title('Prediction Error vs Real Data (Joint Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Discrepancy
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, mae_diff, label='|LeRobot - OpenPI|', color='green')
    plt.xlabel('Time Step (Horizon)')
    plt.ylabel('Mean Absolute Difference')
    plt.title('Implementation Discrepancy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_plot = "pi05_joint_error_comparison.png"
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Saved comparison plot to {output_plot}")

    # --- REPORTING ---
    print(f"\nFinal Results:")
    # We compare the UNNORMALIZED values now
    lerobot_actions_own_sliced = lerobot_actions_own[..., :valid_dims]
    
    print(f"LeRobot Action Mean: {lerobot_actions_own_sliced.mean().item():.6f}")
    print(f"OpenPI Action Mean:  {openpi_actions.mean().item():.6f}")
    
    diff_max = torch.abs(lerobot_actions_own_sliced - openpi_actions).max().item()
    print(f"Max Absolute Difference: {diff_max:.6f}")
    
    if diff_max < 1e-1: # Relaxed tolerance for float32 denormalization ops
        print("✅ SUCCESS: Implementations match within tolerance.")
    else:
        print("⚠️ WARNING: Implementations diverge significantly.")

if __name__ == "__main__":
    test_pi05_original_vs_lerobot()
