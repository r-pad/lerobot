#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
Pre- and post-processing callables for PI0.5 policy.

Replaces the EnvTransition / ProcessorStep / Pipeline abstraction with
two simple callable classes that operate directly on batch dicts / tensors.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.common.constants import (
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_pi05 import pad_vector
from lerobot.common.policies.pi05.preprocessing import Normalizer
from lerobot.common.utils.import_utils import _transformers_available
from lerobot.common.utils.utils import get_safe_torch_device

if _transformers_available:
    from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_batch_dim_obs(batch: dict[str, Any]) -> dict[str, Any]:
    """Add batch dimension (unsqueeze(0)) to unbatched observation tensors."""
    for state_key in [OBS_STATE, OBS_ENV_STATE]:
        if state_key in batch:
            v = batch[state_key]
            if isinstance(v, Tensor) and v.dim() == 1:
                batch[state_key] = v.unsqueeze(0)
    if OBS_IMAGE in batch:
        v = batch[OBS_IMAGE]
        if isinstance(v, Tensor) and v.dim() == 3:
            batch[OBS_IMAGE] = v.unsqueeze(0)
    for key in list(batch.keys()):
        if key.startswith(f"{OBS_IMAGES}."):
            v = batch[key]
            if isinstance(v, Tensor) and v.dim() == 3:
                batch[key] = v.unsqueeze(0)
    # task string → list
    if "task" in batch and isinstance(batch["task"], str):
        batch["task"] = [batch["task"]]
    return batch


def _move_to_device(batch: dict[str, Any], device: torch.device, non_blocking: bool) -> dict[str, Any]:
    """Move all tensors in a flat dict to *device*."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            # Multi-GPU: keep tensor on its existing GPU if both are CUDA
            if v.is_cuda and device.type == "cuda":
                target = v.device
            else:
                target = device
            if target.type == "mps" and v.dtype == torch.float64:
                v = v.to(dtype=torch.float32)
            if v.device != target:
                v = v.to(target, non_blocking=non_blocking)
            out[k] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Pi05Preprocessor
# ---------------------------------------------------------------------------


class Pi05Preprocessor:
    """Callable that turns a raw batch dict into a model-ready batch dict.

    Pipeline (in order):
      1. Add batch dimension to unbatched tensors
      2. Normalize state and action using dataset statistics
      3. Pad state → max_state_dim, discretize, build prompt string
      4. Tokenize prompt with PaliGemma tokenizer
      5. Move everything to target device
    """

    def __init__(self, config: PI05Config, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        self.config = config
        self.normalizer = Normalizer(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )

        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is required. "
                "Install with `pip install 'lerobot[transformers-dep]'`."
            )
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.device = get_safe_torch_device(config.device) if config.device else torch.device("cpu")
        self.non_blocking = "cuda" in str(self.device)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)  # shallow copy

        # 1. Add batch dimension
        batch = _add_batch_dim_obs(batch)

        # 2. Normalize observations + action (in-place on batch keys)
        batch = self.normalizer.normalize_batch(batch)

        # 3. Prepare state prompt (pad, discretize, format)
        batch = self._prepare_state_prompt(batch)

        # 4. Tokenize
        batch = self._tokenize(batch)

        # 5. Move to device
        batch = _move_to_device(batch, self.device, self.non_blocking)

        return batch

    def _prepare_state_prompt(self, batch: dict[str, Any]) -> dict[str, Any]:
        state = batch.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = batch.get("task")
        if tasks is None:
            raise ValueError("No task found in batch")

        state = deepcopy(state)
        state = pad_vector(state, self.config.max_state_dim)

        # State already normalized to [-1, 1]; discretize into 256 bins
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

        batch["task"] = full_prompts
        return batch

    def _tokenize(self, batch: dict[str, Any]) -> dict[str, Any]:
        tasks = batch.get("task")
        if tasks is None:
            raise ValueError("No task found in batch for tokenization")
        if isinstance(tasks, str):
            tasks = [tasks]

        tokenized = self.tokenizer(
            tasks,
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )

        # Detect device from existing tensors for consistency
        target_device = None
        for v in batch.values():
            if isinstance(v, Tensor):
                target_device = v.device
                break

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"].to(dtype=torch.bool)
        if target_device is not None:
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)

        batch[OBS_LANGUAGE_TOKENS] = input_ids
        batch[OBS_LANGUAGE_ATTENTION_MASK] = attention_mask
        return batch


# ---------------------------------------------------------------------------
# Pi05Postprocessor
# ---------------------------------------------------------------------------


class Pi05Postprocessor:
    """Callable that unnormalizes and moves a policy action tensor to CPU.

    Pipeline:
      1. Unnormalize action
      2. Move to CPU
    """

    def __init__(self, config: PI05Config, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        self.unnormalizer = Normalizer(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )

    def __call__(self, action: Tensor) -> Tensor:
        action = self.unnormalizer.unnormalize_action(action)
        return action.cpu()


# ---------------------------------------------------------------------------
# Public factory (backward-compatible signature)
# ---------------------------------------------------------------------------


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
) -> tuple[Pi05Preprocessor, Pi05Postprocessor]:
    """Constructs pre-processor and post-processor for the PI0.5 policy."""
    return (
        Pi05Preprocessor(config, dataset_stats),
        Pi05Postprocessor(config, dataset_stats),
    )
