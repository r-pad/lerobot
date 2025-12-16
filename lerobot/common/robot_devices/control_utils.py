# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

########################################################################################
# Utilities
########################################################################################


import logging
import os
import time
import traceback
from contextlib import nullcontext
from copy import copy, deepcopy
from functools import cache
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot, load_stats, cast_stats_to_numpy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import prepare_observation_for_inference
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer
from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

# OpenPI imports (optional, for PI05 inference)
try:
    from openpi.models_pytorch import preprocessing_pytorch as openpi_preprocessing
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from transformers import AutoTokenizer
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False
    openpi_preprocessing = None
    PI0Pytorch = None
    AutoTokenizer = None

# Aloha policy helpers (for state/action transformations)
try:
    from aloha_policy import (
        _decode_state,
        _encode_actions_inv,
        normalize_state_mean_std,
        unnormalize_action_mean_std,
    )
    ALOHA_POLICY_AVAILABLE = True
except ImportError:
    ALOHA_POLICY_AVAILABLE = False
    _decode_state = None
    _encode_actions_inv = None
    normalize_state_mean_std = None
    unnormalize_action_mean_std = None

# KEEP_INDICES for filtering state/actions to joint angles (matching test_pi05_lerobot_mug_bin.py)
KEEP_INDICES = [0, 1, 3, 6, 5, 7, 8, 9, 10, 12, 15, 14, 16, 17]

def add_eef_pose(real_joints):
    eef_pose, eef_pose_se3 = forward_kinematics(ALOHA_CONFIGURATION, real_joints)
    eef_pose = torch.cat([eef_pose, real_joints[-1][None]], axis=0).float()
    return eef_pose

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    # logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


# Cache for OpenPI model and tokenizer
_OPENPI_MODEL_CACHE: dict[str, tuple] = {}
_OPENPI_TOKENIZER_CACHE = None


def _load_openpi_model(policy: PreTrainedPolicy | None = None, pretrained_path: str | Path | None = None, policy_config: dict | None = None):
    """Load the original OpenPI model (PI0Pytorch) for inference.
    
    Args:
        policy: LeRobot policy (optional, can be None if using OpenPI only)
        pretrained_path: Path to checkpoint directory
        policy_config: Policy config dict (used when policy is None)
    """
    if not OPENPI_AVAILABLE:
        raise ImportError("OpenPI is not available. Please install it to use OpenPI inference.")
    
    cache_key = str(pretrained_path) if pretrained_path else "default"
    if cache_key in _OPENPI_MODEL_CACHE:
        return _OPENPI_MODEL_CACHE[cache_key]
    
    # Create config matching OpenPI structure
    # Get config from policy if available, otherwise use defaults or policy_config
    if policy is not None:
        max_action_dim = getattr(policy.config, "max_action_dim", 32)
        chunk_size = getattr(policy.config, "chunk_size", 50)
        paligemma_variant = getattr(policy.config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(policy.config, "action_expert_variant", "gemma_300m")
        dtype = getattr(policy.config, "dtype", "float32")
        device = getattr(policy.config, "device", "cpu")
    elif policy_config is not None:
        max_action_dim = policy_config.get("max_action_dim", 32)
        chunk_size = policy_config.get("chunk_size", 50)
        paligemma_variant = policy_config.get("paligemma_variant", "gemma_2b")
        action_expert_variant = policy_config.get("action_expert_variant", "gemma_300m")
        dtype = policy_config.get("dtype", "float32")
        device = policy_config.get("device", "cpu")
    else:
        # Defaults - try to load from checkpoint config if available
        max_action_dim = 32
        chunk_size = 50
        paligemma_variant = "gemma_2b"
        action_expert_variant = "gemma_300m"
        dtype = "float32"
        device = "cpu"
        
        # Try to load config from checkpoint
        if pretrained_path:
            pretrained_path = Path(pretrained_path)
            config_file = pretrained_path / "config.json"
            if config_file.exists():
                import json
                with open(config_file) as f:
                    checkpoint_config = json.load(f)
                max_action_dim = checkpoint_config.get("max_action_dim", max_action_dim)
                chunk_size = checkpoint_config.get("chunk_size", chunk_size)
                paligemma_variant = checkpoint_config.get("paligemma_variant", paligemma_variant)
                action_expert_variant = checkpoint_config.get("action_expert_variant", action_expert_variant)
                dtype = checkpoint_config.get("dtype", dtype)
                device = checkpoint_config.get("device", device)
    
    # Create config object with the determined values
    # Use a simple class with attributes set from outer scope variables
    class PI05BaseOriginalConfig:
        pass
    
    config = PI05BaseOriginalConfig()
    config.action_dim = max_action_dim
    config.action_horizon = chunk_size
    config.paligemma_variant = paligemma_variant
    config.action_expert_variant = action_expert_variant
    config.precision = dtype
    config.pi05 = True
    config.dtype = dtype
    
    openpi_model = PI0Pytorch(config)
    
    # Load weights if pretrained_path is provided
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        model_file = pretrained_path / "model.safetensors"
        
        if model_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
            
            # Unwrap 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            openpi_model.load_state_dict(new_state_dict, strict=False)
            logging.info(f"Loaded OpenPI weights from {model_file}")
        else:
            logging.warning(f"OpenPI model file not found at {model_file}, using uninitialized model")
    
    # Load model on GPU (device)
    device = get_safe_torch_device(device)
    openpi_model.to(device)
    openpi_model.eval()
    
    _OPENPI_MODEL_CACHE[cache_key] = openpi_model
    return openpi_model


def _get_openpi_tokenizer():
    """Get or create the OpenPI tokenizer."""
    global _OPENPI_TOKENIZER_CACHE
    if _OPENPI_TOKENIZER_CACHE is None:
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI is not available. Please install it to use OpenPI inference.")
        tokenizer_name = "google/paligemma-3b-pt-224"
        _OPENPI_TOKENIZER_CACHE = AutoTokenizer.from_pretrained(tokenizer_name)
    return _OPENPI_TOKENIZER_CACHE


def _preprocess_observation_for_openpi(
    observation: dict[str, torch.Tensor],
    policy: PreTrainedPolicy | None,
    dataset_stats: dict | None = None,
    keep_indices: list[int] | None = None,
    policy_config: dict | None = None,
) -> dict:
    """Preprocess observation for OpenPI inference using OpenPI's preprocessing."""
    if not OPENPI_AVAILABLE or not ALOHA_POLICY_AVAILABLE:
        raise ImportError("OpenPI and aloha_policy are required for OpenPI inference.")
    
    batch_size = 1
    device = observation[OBS_STATE].device
    
    # Get tokenizer
    tokenizer = _get_openpi_tokenizer()
    
    # Get task
    task = observation.get("task", "Pick up the object")
    if isinstance(task, str):
        tasks = [task]
    else:
        tasks = [task] if not isinstance(task, (list, tuple)) else list(task)
    
    # Process state
    state = observation[OBS_STATE]
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    # Filter to joint angles if keep_indices provided
    if keep_indices is not None and state.shape[-1] >= len(keep_indices):
        state = state[:, keep_indices]
    
    state = deepcopy(state)
    
    # Normalize state if stats available
    if dataset_stats:
        stats_root = dataset_stats.get("norm_stats", dataset_stats)
        if "state" in stats_root:
            state_mean = torch.tensor(stats_root["state"]["mean"], device=device, dtype=torch.float32)
            state_std = torch.tensor(stats_root["state"]["std"], device=device, dtype=torch.float32)
            
            # Filter stats if needed
            if keep_indices is not None and state_mean.shape[0] >= len(keep_indices):
                state_mean = state_mean[keep_indices]
                state_std = state_std[keep_indices]
            
            # Transform Aloha -> PI
            state_np = state.cpu().numpy()
            decoded = []
            for s in state_np:
                decoded.append(_decode_state(s, adapt_to_pi=True))
            state = torch.tensor(np.stack(decoded), device=device, dtype=torch.float32)
            
            # Normalize
            state = normalize_state_mean_std(state, state_mean, state_std)
    
    # Pad state to max_state_dim
    if policy is not None:
        max_state_dim = getattr(policy.config, "max_state_dim", state.shape[-1])
    elif policy_config is not None:
        max_state_dim = policy_config.get("max_state_dim", state.shape[-1])
    else:
        max_state_dim = state.shape[-1]
    from lerobot.common.policies.pi05.modeling_pi05 import pad_vector
    state = pad_vector(state, max_state_dim)
    
    # Discretize state
    state_np = state.cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    
    # Create prompts
    full_prompts = []
    for i, task_str in enumerate(tasks):
        cleaned_text = str(task_str).strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized_states[i]))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        full_prompts.append(full_prompt)
    
    # Tokenize
    if policy is not None:
        max_token_len = getattr(policy.config, "tokenizer_max_length", 200)
    elif policy_config is not None:
        max_token_len = policy_config.get("tokenizer_max_length", 200)
    else:
        max_token_len = 200
    tokenized = tokenizer(
        full_prompts,
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=max_token_len,
        return_tensors="pt",
    )
    
    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].to(device, dtype=torch.bool)
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)
    
    # Process images - convert from LeRobot [0,1] to OpenPI [-1,1] and resize to 224x224
    image_dict = {}
    image_masks_dict = {}
    
    # Map camera names - try to find images in observation
    camera_mapping = {
        "base_0_rgb": ["observation.images.base_0_rgb", "observation.images.cam_azure_kinect_front"],
        "left_wrist_0_rgb": ["observation.images.left_wrist_0_rgb", "observation.images.cam_azure_kinect_back"],
        "right_wrist_0_rgb": ["observation.images.right_wrist_0_rgb", "observation.images.cam_wrist"],
    }
    
    for target_key, possible_keys in camera_mapping.items():
        img_tensor = None
        for key in possible_keys:
            if key in observation:
                img_tensor = observation[key]
                break
        
        if img_tensor is not None:
            # Ensure shape is [C, H, W]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.shape[1] != 3:  # Ensure channel first
                img_tensor = img_tensor.permute(0, 3, 1, 2)
            # Resize to 224x224 (OpenPI expectation)
            img_tensor = TF.resize(img_tensor, [224, 224], antialias=True)
            # Convert [0,1] to [-1,1]
            image_dict[target_key] = img_tensor * 2.0 - 1.0
            image_masks_dict[target_key] = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            # Create zero image if not found
            image_dict[target_key] = torch.zeros(batch_size, 3, 224, 224, device=device) - 1.0
            image_masks_dict[target_key] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Create PI05Observation-like structure
    class PI05Observation:
        def __init__(self, state, images, image_masks, tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask):
            self.state = state
            self.images = images
            self.image_masks = image_masks
            self.tokenized_prompt = tokenized_prompt
            self.tokenized_prompt_mask = tokenized_prompt_mask
            self.token_ar_mask = token_ar_mask
            self.token_loss_mask = token_loss_mask
    
    raw_observation = PI05Observation(
        state=state,
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )
    
    # Use OpenPI preprocessing
    processed_obs = openpi_preprocessing.preprocess_observation_pytorch(raw_observation, train=False)
    return processed_obs


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy | None,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, any], dict[str, any]] | None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
    use_openpi_inference: bool = False,
    pretrained_path: str | Path | None = None,
    dataset_stats: dict | None = None,
    policy_config: dict | None = None,
):
    """Predict action using either LeRobot or OpenPI inference.
    
    Args:
        policy: LeRobot policy (can be None if using OpenPI inference only)
        use_openpi_inference: If True and policy is PI05, use OpenPI inference instead of LeRobot.
        pretrained_path: Path to pretrained model checkpoint (for OpenPI model loading).
        dataset_stats: Dataset statistics for normalization (required for OpenPI inference).
        policy_config: Policy config dict (used when policy is None).
    """
    # Check if we should use OpenPI inference
    if use_openpi_inference:
        if not OPENPI_AVAILABLE or not ALOHA_POLICY_AVAILABLE:
            logging.warning("OpenPI or aloha_policy not available, falling back to LeRobot inference")
            use_openpi_inference = False
        elif policy is not None and getattr(policy.config, "type", None) != "pi05":
            logging.warning("Policy is not PI05, disabling OpenPI inference")
            use_openpi_inference = False
    
    if use_openpi_inference:
        # Use OpenPI inference
        observation = copy(observation)
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        
        # Load OpenPI model (policy can be None when using OpenPI only)
        openpi_model = _load_openpi_model(policy, pretrained_path, policy_config)
        
        # Preprocess for OpenPI
        keep_indices = KEEP_INDICES  # Filter to joint angles (matching test_pi05_lerobot_mug_bin.py)
        # Create a minimal policy-like object for preprocessing if policy is None
        if policy is None:
            # Create a minimal config object for preprocessing
            class MinimalPolicyConfig:
                def __init__(self, config_dict):
                    for k, v in config_dict.items():
                        setattr(self, k, v)
                    self.type = "pi05"
            
            class MinimalPolicy:
                def __init__(self, config_dict):
                    self.config = MinimalPolicyConfig(config_dict or {})
                    self.action_space = "joint"
            
            minimal_policy = MinimalPolicy(policy_config)
            processed_obs = _preprocess_observation_for_openpi(
                observation, minimal_policy, dataset_stats, keep_indices, policy_config
            )
        else:
            processed_obs = _preprocess_observation_for_openpi(
                observation, policy, dataset_stats, keep_indices, policy_config
            )
        
        # Sample actions using OpenPI
        batch_size = observation[OBS_STATE].shape[0]
        if policy is not None:
            action_horizon = getattr(policy.config, "chunk_size", 50)
            action_dim = getattr(policy.config, "max_action_dim", 32)
        else:
            action_horizon = policy_config.get("chunk_size", 50) if policy_config else 50
            action_dim = policy_config.get("max_action_dim", 32) if policy_config else 32
        
        # Get the device the model is actually on
        model_device = next(openpi_model.parameters()).device
        
        # Move all observation tensors to model device to avoid device mismatch
        if hasattr(processed_obs, 'images'):
            processed_obs.images = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in processed_obs.images.items()
            }
        if hasattr(processed_obs, 'image_masks'):
            processed_obs.image_masks = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in processed_obs.image_masks.items()
            }
        if hasattr(processed_obs, 'tokenized_prompt'):
            processed_obs.tokenized_prompt = processed_obs.tokenized_prompt.to(model_device)
        if hasattr(processed_obs, 'tokenized_prompt_mask'):
            processed_obs.tokenized_prompt_mask = processed_obs.tokenized_prompt_mask.to(model_device)
        if hasattr(processed_obs, 'token_ar_mask'):
            processed_obs.token_ar_mask = processed_obs.token_ar_mask.to(model_device)
        if hasattr(processed_obs, 'token_loss_mask'):
            processed_obs.token_loss_mask = processed_obs.token_loss_mask.to(model_device)
        
        noise_shape = (batch_size, action_horizon, action_dim)
        noise = torch.randn(noise_shape, dtype=torch.float32, device=model_device)
        
        with torch.inference_mode():
            # Call OpenPI's sample_actions with the processed observation
            # The original OpenPI PI0Pytorch.sample_actions takes: device, observation, noise, num_steps
            openpi_actions = openpi_model.sample_actions(
                device=model_device,
                observation=processed_obs,
                noise=noise,
                num_steps=10,
            )
        
        # Post-process actions following test_pi05_lerobot_mug_bin.py logic:
        # 1. Slice to 14 dims (len(KEEP_INDICES))
        # 2. Filter stats with KEEP_INDICES
        # 3. Unnormalize
        # 4. Transform PI -> Aloha
        
        # Ensure openpi_actions is a regular tensor (not inference-mode) before processing
        if openpi_actions.is_inference():
            openpi_actions = openpi_actions.clone().detach()
        
        # 1. Slice to 14 dims (matching test_pi05_lerobot_mug_bin.py line 587)
        openpi_actions = openpi_actions[..., :len(KEEP_INDICES)]
        
        # Unnormalize actions if stats available
        if dataset_stats:
            stats_root = dataset_stats.get("norm_stats", dataset_stats)
            action_stats = stats_root.get("action") or stats_root.get("actions")
            if action_stats:
                action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.float32)
                action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.float32)
                
                # 2. Filter stats with KEEP_INDICES, guarding against shorter stats tensors
                # if action_mean.shape[0] >= len(KEEP_INDICES):
                #     valid_keep = KEEP_INDICES
                # else:
                #     valid_keep = [i for i in KEEP_INDICES if i < action_mean.shape[0]]
                # if len(valid_keep) > 0:
                #     breakpoint()
                #     action_mean = action_mean[valid_keep]
                #     action_std = action_std[valid_keep]
                #     openpi_actions = openpi_actions[..., :len(valid_keep)]
                # else:
                #     # Fallback: slice to available dims to avoid device-side assert
                #     openpi_actions = openpi_actions[..., :action_mean.shape[0]]
                
                # 3. Unnormalize (matching test_pi05_lerobot_mug_bin.py line 599)
                openpi_actions = unnormalize_action_mean_std(openpi_actions, action_mean, action_std)
                
                # 4. Transform PI -> Aloha (matching test_pi05_lerobot_mug_bin.py lines 601-606)
                openpi_np = openpi_actions.cpu().numpy()
                encoded = []
                for b in range(openpi_np.shape[0]):
                    encoded.append(_encode_actions_inv(openpi_np[b], adapt_to_pi=True))
                openpi_actions = torch.tensor(np.stack(encoded), device=device, dtype=torch.float32)
        
        # Expand filtered 14-dim action back to full 18-dim action
        # Pad missing indices: 2<-1, 4<-3, 11<-10, 13<-12
        if openpi_actions.shape[-1] == len(KEEP_INDICES):
            # Get rest state for filling any remaining missing indices (shouldn't be needed now)
            from lerobot.common.utils.aloha_utils import ALOHA_REST_STATE
            rest_state = ALOHA_REST_STATE[0].to(device)  # Shape: [18]
            
            # Create full 18-dim action tensor
            batch_size, horizon = openpi_actions.shape[:2]
            full_actions = torch.zeros(batch_size, horizon, 18, device=device, dtype=openpi_actions.dtype)
            
            # Fill with rest state values first (will be overwritten by filtered values and padding)
            full_actions[:] = rest_state.unsqueeze(0).unsqueeze(0)
            
            # Create a mapping from original index to filtered index position
            # KEEP_INDICES maps: filtered[i] -> original[KEEP_INDICES[i]]
            # We need to reverse this: original[orig_idx] -> filtered[filtered_pos]
            orig_to_filtered = {orig_idx: i for i, orig_idx in enumerate(KEEP_INDICES)}
            
            # Place filtered values at their original positions (reversing the KEEP_INDICES mapping)
            for orig_idx in sorted(KEEP_INDICES):  # Process in sorted order to ensure correct placement
                filtered_pos = orig_to_filtered[orig_idx]
                full_actions[:, :, orig_idx] = openpi_actions[:, :, filtered_pos]
            
            # Pad missing indices as requested:
            # Index 2 gets value from index 1
            # Index 4 gets value from index 3
            # Index 11 gets value from index 10
            # Index 13 gets value from index 12
            # Find the filtered indices that correspond to the source indices
            idx_1_in_filtered = orig_to_filtered[1]   # Position in filtered array for original index 1
            idx_3_in_filtered = orig_to_filtered[3]   # Position in filtered array for original index 3
            idx_10_in_filtered = orig_to_filtered[10]  # Position in filtered array for original index 10
            idx_12_in_filtered = orig_to_filtered[12]  # Position in filtered array for original index 12
            
            full_actions[:, :, 2] = openpi_actions[:, :, idx_1_in_filtered]   # Use value from index 1
            full_actions[:, :, 4] = openpi_actions[:, :, idx_3_in_filtered]   # Use value from index 3
            full_actions[:, :, 11] = openpi_actions[:, :, idx_10_in_filtered] # Use value from index 10
            full_actions[:, :, 13] = openpi_actions[:, :, idx_12_in_filtered] # Use value from index 12
            
            openpi_actions = full_actions
        
        # Get first action from horizon
        action = openpi_actions[0, 0]  # [batch, horizon, dim] -> [dim]
        
        # Get EEF action if needed (use robot adapter)
        action_eef = None
        robot_adapter = None
        if policy is not None and hasattr(policy, "robot_adapter"):
            robot_adapter = policy.robot_adapter
        elif policy_config is not None:
            # Create robot adapter from policy_config when policy is None
            robot_type = policy_config.get("robot_type", "aloha")
            action_space = policy_config.get("action_space", "joint")
            if robot_type == "aloha":
                from lerobot.common.policies.robot_adapters import AlohaAdapter
                robot_adapter = AlohaAdapter(action_space)
            elif robot_type == "libero_franka":
                from lerobot.common.policies.robot_adapters import LiberoFrankaAdapter
                obs_key = "observation.state"
                act_key = "action"
                robot_adapter = LiberoFrankaAdapter(obs_key, act_key)
        
        if robot_adapter is not None:
            # Clone action to allow in-place modifications in transform_action
            # (openpi_actions is already converted to regular tensor above)
            action_clone = action.clone()
            state_tensor = observation[OBS_STATE]
            action = robot_adapter.transform_action(action_clone.unsqueeze(0), state_tensor.unsqueeze(0)).squeeze(0)
            action_eef = robot_adapter.get_eef_action(action_clone.unsqueeze(0)).squeeze(0)
        
        print(f"Action: {action}")
        # Move to cpu
        action = action.to("cpu")
        if action_eef is not None:
            action_eef = action_eef.to("cpu")
        else:
            action_eef = torch.zeros_like(action)
        
        return action, action_eef
    
    # Default LeRobot inference
    if policy is None:
        raise ValueError("Policy is None and OpenPI inference is disabled. Cannot predict action.")
    
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
                # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)

        # Compute the next action with the policy
        # based on the current observation
        action, action_eef = policy.select_action(observation, preprocessor, postprocessor)

        # Remove batch dimension
        action, action_eef = action.squeeze(0), action_eef.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")
        action_eef = action_eef.to("cpu")

        return action, action_eef


_PI05_DEFAULT_TOKENIZER = "google/paligemma-3b-pt-224"
_LANGUAGE_TOKENIZER_CACHE: dict[tuple[str, int, str, str, bool], TokenizerProcessorStep] = {}


def _get_language_tokenizer(
    tokenizer_name: str,
    max_length: int,
    padding_side: str = "right",
    padding: str = "max_length",
    truncation: bool = True,
) -> TokenizerProcessorStep:
    cache_key = (tokenizer_name, max_length, padding_side, padding, truncation)
    tokenizer = _LANGUAGE_TOKENIZER_CACHE.get(cache_key)
    if tokenizer is None:
        tokenizer = TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            padding_side=padding_side,
            padding=padding,
            truncation=truncation,
        )
        _LANGUAGE_TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _pad_state_for_language_tokens(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    if state.shape[-1] >= target_dim:
        return state
    pad_size = target_dim - state.shape[-1]
    return F.pad(state, (0, pad_size))


def _discretize_state(state: torch.Tensor) -> np.ndarray:
    state_np = state.detach().cpu().numpy()
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discretized = np.digitize(state_np, bins, right=False) - 1
    return np.clip(discretized, 0, 255).astype(int)


def _compose_language_prompts(tasks: list[str], discretized_states: np.ndarray) -> list[str]:
    prompts: list[str] = []
    num_states = discretized_states.shape[0]
    for idx, task in enumerate(tasks):
        state_idx = idx if idx < num_states else num_states - 1
        cleaned_task = str(task).strip().replace("_", " ").replace("\n", " ")
        state_tokens = " ".join(str(val) for val in discretized_states[state_idx])
        prompts.append(f"Task: {cleaned_task}, State: {state_tokens};\nAction: ")
    return prompts


def maybe_add_language_tokens(observation: dict, policy: PreTrainedPolicy | None) -> None:
    if policy is None:
        return

    if OBS_LANGUAGE_TOKENS in observation and OBS_LANGUAGE_ATTENTION_MASK in observation:
        return

    if getattr(policy.config, "type", None) != "pi05":
        return

    task_value = observation.get("task")
    if task_value is None:
        raise ValueError(
            "PI05 policies require a task description. Please set `control.single_task` when running the controller."
        )

    if isinstance(task_value, str):
        tasks = [task_value]
    elif isinstance(task_value, (list, tuple)):
        if not task_value:
            raise ValueError("Task list cannot be empty for PI05 policies.")
        tasks = list(task_value)
    else:
        raise TypeError("Task must be a string or a list of strings when running a PI05 policy.")

    if len(tasks) != 1:
        raise ValueError(
            f"PI05 inference currently supports exactly one task string per call. Got {len(tasks)} tasks."
        )

    state_value = observation.get(OBS_STATE)
    if state_value is None:
        raise ValueError("`observation.state` is required to build language tokens for PI05 policies.")

    state_tensor = torch.as_tensor(state_value, dtype=torch.float32)
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)
    elif state_tensor.dim() != 2:
        raise ValueError(
            "`observation.state` must be a 1D or 2D tensor to build language tokens for PI05 policies."
        )

    max_state_dim = getattr(policy.config, "max_state_dim", state_tensor.shape[-1])
    padded_state = _pad_state_for_language_tokens(state_tensor, max_state_dim)
    discretized_states = _discretize_state(padded_state)
    prompts = _compose_language_prompts(tasks, discretized_states)

    tokenizer_name = getattr(policy.config, "tokenizer_name", _PI05_DEFAULT_TOKENIZER)
    max_length = getattr(policy.config, "tokenizer_max_length", 512)
    tokenizer = _get_language_tokenizer(tokenizer_name, max_length)
    tokenized_prompt = tokenizer._tokenize_text(prompts)

    if tokenized_prompt["input_ids"].shape[0] != 1:
        raise ValueError("PI05 inference currently only supports batch size 1 when building language tokens.")

    observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"].squeeze(0)
    observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].squeeze(0).to(dtype=torch.bool)


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
    preprocessor,
    postprocessor,
    use_openpi_inference: bool = False,
    pretrained_path: str | Path | None = None,
    dataset_stats: dict | None = None,
    policy_config: dict | None = None,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None and not use_openpi_inference,
        single_task=single_task,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_openpi_inference=use_openpi_inference,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_stats,
        policy_config=policy_config,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    preprocessor=None,
    postprocessor=None,
    use_openpi_inference: bool = False,
    pretrained_path: str | Path | None = None,
    dataset_stats: dict | None = None,
    policy_config: dict | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()

    # Controls starts, if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
            if robot.use_eef:
                observation["observation.right_eef_pose"] = add_eef_pose(observation['observation.state'])
                action["action.right_eef_pose"] = add_eef_pose(action['action'])
        else:
            observation = robot.capture_observation()
            if robot.use_eef:
                observation["observation.right_eef_pose"] = add_eef_pose(observation['observation.state'])
            action = None

            # Use policy or OpenPI inference
            if policy is not None or use_openpi_inference:
                # Pretty ugly, but moving this code inside the policy makes it uglier to visualize
                # the goal_gripper_proj key.

                # Goal conditioning only works with LeRobot policy (not OpenPI)
                if policy is not None and hasattr(policy.config, "enable_goal_conditioning") and policy.config.enable_goal_conditioning:
                    # Generate new goal prediction when queue is empty
                    # This code is specific to diffusion policy / kinect :(
                    if hasattr(policy, "_queues") and len(policy._queues[policy.act_key]) == 0:
                        # Gather observations from all configured cameras
                        camera_obs = {}
                        state = observation["observation.state"].numpy()

                        # Setup renderer once for all cameras if using phantomize
                        if hasattr(policy.config, "phantomize") and policy.config.phantomize:
                            if policy.renderer is None:
                                intrinsics_txts, extrinsics_txts, virtual_camera_names = [], [], []
                                for cam_name in policy.high_level.camera_names:
                                    intrinsics_txts.append(f"lerobot/scripts/{policy.high_level.calibration_data[cam_name]['intrinsics']}")
                                    extrinsics_txts.append(f"lerobot/scripts/{policy.high_level.calibration_data[cam_name]['extrinsics']}")
                                    virtual_camera_names.append(VIRTUAL_CAMERA_MAPPING[cam_name])

                                # Get image dimensions from first camera
                                first_cam = policy.high_level.camera_names[0]
                                rgb_key = f"observation.images.{first_cam}.color"
                                height, width, _ = observation[rgb_key].numpy().shape

                                # Setup renderer with all cameras at once
                                policy.renderer = setup_renderer(
                                    ALOHA_MODEL,
                                    intrinsics_txts,
                                    extrinsics_txts,
                                    policy.downsample_factor,
                                    width,
                                    height,
                                    virtual_camera_names
                                )

                        # Gather camera observations and apply phantomize if needed
                        for cam_name in policy.high_level.camera_names:
                            rgb_key = f"observation.images.{cam_name}.color"
                            depth_key = f"observation.images.{cam_name}.transformed_depth"

                            # Validate camera data exists
                            if rgb_key not in observation or depth_key not in observation:
                                raise ValueError(
                                    f"Required camera observation '{cam_name}' not found. "
                                    f"Available keys: {policy.high_level.camera_names}"
                                )
                            camera_obs[cam_name] = {
                                "rgb": observation[rgb_key].numpy(),
                                "depth": observation[depth_key].numpy().squeeze()
                            }

                            if hasattr(policy.config, "phantomize") and policy.config.phantomize:
                                # Overlay RGB with rendered robot
                                render = render_and_overlay(
                                    policy.renderer,
                                    ALOHA_MODEL,
                                    state,
                                    camera_obs[cam_name]["rgb"].copy(),
                                    policy.downsample_factor,
                                    VIRTUAL_CAMERA_MAPPING[cam_name],
                                )
                                camera_obs[cam_name]["rgb"] = render
                        # import matplotlib.pyplot as plt, cv2, os
                        # os.makedirs("phantomize_inference_viz", exist_ok=True)
                        # img = cv2.hconcat([camera_obs[cam_name]["rgb"] for cam_name in policy.high_level.camera_names])
                        # plt.imsave(f"phantomize_inference_viz/{time.time()}.png", img)


                        # Get dict of projections for all cameras
                        gripper_projs = policy.high_level.predict_and_project(
                            single_task, camera_obs,
                            robot_type=policy.config.robot_type,
                            robot_kwargs={"observation.state": observation["observation.state"]}
                        )  # Returns dict[str, np.ndarray]

                        # Store as dict of tensors
                        for cam_name, proj in gripper_projs.items():
                            policy.latest_gripper_proj[cam_name] = torch.from_numpy(proj)

                    # Add goal projection to each camera observation
                    for cam_name in policy.high_level.camera_names:
                        observation[f"observation.images.{cam_name}.goal_gripper_proj"] = policy.latest_gripper_proj[cam_name]

                observation["task"] = single_task
                observation_for_policy = copy(observation)
                
                # Only add language tokens if using LeRobot inference (OpenPI handles its own tokenization)
                if policy is not None and not use_openpi_inference:
                    maybe_add_language_tokens(observation_for_policy, policy)
                
                # Get device from policy or policy_config
                if policy is not None:
                    device = get_safe_torch_device(policy.config.device)
                    use_amp = policy.config.use_amp
                elif policy_config is not None:
                    device = get_safe_torch_device(policy_config.get("device", "cpu"))
                    use_amp = False
                else:
                    device = get_safe_torch_device("cpu")
                    use_amp = False
                
                pred_action, pred_action_eef = predict_action(
                    observation_for_policy,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                    use_openpi_inference=use_openpi_inference,
                    pretrained_path=pretrained_path,
                    dataset_stats=dataset_stats,
                    policy_config=policy_config,
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                breakpoint()
                action = robot.send_action(pred_action)
                action = {"action": action}
                if robot.use_eef:
                    action["action.right_eef_pose"] = pred_action_eef

        if dataset is not None:
            # Ensure action is not None (should be set by policy or teleoperation)
            if action is None:
                logging.warning("Action is None, skipping frame")
                continue
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

            # Add point cloud visualization from high-level model
            if policy is not None and hasattr(policy, 'high_level'):
                hl_wrapper = policy.high_level

                white_bg = False
                if white_bg:
                    # Set white background for 3D view using blueprint
                    blueprint = rr.blueprint.Blueprint(
                        rr.blueprint.Spatial3DView(
                            origin="high_level",
                            background=[255, 255, 255]  # White background
                        )
                    )
                    rr.send_blueprint(blueprint)

                if hl_wrapper.last_pcd_xyz is not None:
                    pcd_rgb = ((hl_wrapper.last_pcd_rgb + 1) * 255 / 2).astype(np.uint8)
                    # Scene point cloud with colors
                    rr.log("high_level/scene_pointcloud", rr.Points3D(hl_wrapper.last_pcd_xyz, colors=pcd_rgb))

                # Gripper point cloud
                if hl_wrapper.last_gripper_pcd is not None:
                    rr.log("high_level/gripper_pointcloud", 
                           rr.Points3D(hl_wrapper.last_gripper_pcd, colors=[255, 0, 0]))

                if hl_wrapper.last_goal_prediction is not None:
                    # Goal prediction
                    rr.log("high_level/goal_prediction",
                        rr.Points3D(hl_wrapper.last_goal_prediction, colors=[0, 255, 0], radii=0.005))

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
