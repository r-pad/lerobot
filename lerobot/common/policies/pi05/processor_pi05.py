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
Self-contained pre/post-processor pipelines for PI05.

Inlines the subset of the upstream processor framework that PI05 actually uses,
avoiding the full framework dependency (registry, serialization, Hub integration, etc.).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import Enum
from functools import singledispatch
from typing import Any, TypeAlias, TypedDict

import numpy as np
import torch
from torch import Tensor

from lerobot.common.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.common.utils.import_utils import _transformers_available, is_package_available
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

if _transformers_available:
    from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Core types (inlined from processor/core.py)
# ---------------------------------------------------------------------------

class TransitionKey(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


PolicyAction: TypeAlias = torch.Tensor
RobotObservation: TypeAlias = dict[str, Any]

EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: RobotObservation | None,
        TransitionKey.ACTION.value: PolicyAction | None,
        TransitionKey.REWARD.value: float | torch.Tensor | None,
        TransitionKey.DONE.value: bool | torch.Tensor | None,
        TransitionKey.TRUNCATED.value: bool | torch.Tensor | None,
        TransitionKey.INFO.value: dict[str, Any] | None,
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,
    },
)


def _create_transition(
    observation: RobotObservation | None = None,
    action: PolicyAction | None = None,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


# ---------------------------------------------------------------------------
# Tensor conversion helpers (inlined from processor/converters.py)
# ---------------------------------------------------------------------------

@singledispatch
def _to_tensor(value: Any, *, dtype=torch.float32, device=None) -> torch.Tensor:
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@_to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kw) -> torch.Tensor:
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@_to_tensor.register(np.ndarray)
def _(value: np.ndarray, *, dtype=torch.float32, device=None, **kw) -> torch.Tensor:
    if value.ndim == 0:
        return torch.tensor(value.item(), dtype=dtype, device=device)
    t = torch.from_numpy(value)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


@_to_tensor.register(int)
@_to_tensor.register(float)
@_to_tensor.register(np.integer)
@_to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kw) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


@_to_tensor.register(list)
@_to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kw) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


@_to_tensor.register(dict)
def _(value: dict, *, device=None, **kw) -> dict:
    if not value:
        return {}
    result = {}
    for key, sub in value.items():
        if sub is None:
            continue
        result[key] = _to_tensor(sub, device=device, **kw)
    return result


def _from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


# ---------------------------------------------------------------------------
# Converters between batch dicts / policy actions and EnvTransition
# ---------------------------------------------------------------------------

def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    subtask_key = {"subtask": batch["subtask"]} if "subtask" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    episode_index_key = {"episode_index": batch["episode_index"]} if "episode_index" in batch else {}
    return {**pad_keys, **task_key, **subtask_key, **index_key, **task_index_key, **episode_index_key}


def _batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    if not isinstance(batch, dict):
        raise ValueError(f"Expected dict, got {type(batch).__name__}")
    obs = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    comp = _extract_complementary_data(batch)
    return _create_transition(
        observation=obs or None,
        action=batch.get(ACTION),
        complementary_data=comp or None,
    )


def _transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    batch: dict[str, Any] = {ACTION: transition.get(TransitionKey.ACTION)}
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp:
        batch.update(comp)
    obs = transition.get(TransitionKey.OBSERVATION)
    if isinstance(obs, dict):
        batch.update(obs)
    return batch


def _policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    return _create_transition(action=action)


def _transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, torch.Tensor):
        raise ValueError(f"Action should be a tensor, got {type(action)}")
    return action


# ---------------------------------------------------------------------------
# Lightweight pipeline: just a list of callables run in sequence
# ---------------------------------------------------------------------------

class PI05ProcessorPipeline:
    """A simple sequential pipeline: convert input → run steps → convert output."""

    def __init__(
        self,
        steps: Sequence[Callable[[EnvTransition], EnvTransition]],
        name: str = "pipeline",
        to_transition: Callable[..., EnvTransition] = _batch_to_transition,
        to_output: Callable[[EnvTransition], Any] = _transition_to_batch,
    ):
        self.steps = list(steps)
        self.name = name
        self.to_transition = to_transition
        self.to_output = to_output

    def __call__(self, data):
        transition = self.to_transition(data)
        for step in self.steps:
            transition = step(transition)
        return self.to_output(transition)


# ---------------------------------------------------------------------------
# Processor steps (inlined, stripped of registry/serialization/ABC overhead)
# ---------------------------------------------------------------------------

def rename_observations_step(rename_map: dict[str, str]) -> Callable[[EnvTransition], EnvTransition]:
    """Renames observation keys according to rename_map."""

    def step(transition: EnvTransition) -> EnvTransition:
        new_t = transition.copy()
        obs = new_t.get(TransitionKey.OBSERVATION)
        if obs is None or not isinstance(obs, dict):
            raise ValueError("Requires observation in transition")
        new_t[TransitionKey.OBSERVATION] = {
            rename_map.get(k, k): v for k, v in obs.items()
        }
        return new_t

    return step


def add_batch_dimension_step() -> Callable[[EnvTransition], EnvTransition]:
    """Adds a batch dimension (unsqueeze(0)) to observations, actions, and complementary data."""

    def step(transition: EnvTransition) -> EnvTransition:
        t = transition.copy()

        # Observations
        obs = t.get(TransitionKey.OBSERVATION)
        if obs is not None and isinstance(obs, dict):
            new_obs = dict(obs)
            for state_key in [OBS_STATE, OBS_ENV_STATE]:
                if state_key in new_obs:
                    v = new_obs[state_key]
                    if isinstance(v, Tensor) and v.dim() == 1:
                        new_obs[state_key] = v.unsqueeze(0)
            if OBS_IMAGE in new_obs:
                v = new_obs[OBS_IMAGE]
                if isinstance(v, Tensor) and v.dim() == 3:
                    new_obs[OBS_IMAGE] = v.unsqueeze(0)
            for k, v in new_obs.items():
                if k.startswith(f"{OBS_IMAGES}.") and isinstance(v, Tensor) and v.dim() == 3:
                    new_obs[k] = v.unsqueeze(0)
            t[TransitionKey.OBSERVATION] = new_obs

        # Action
        action = t.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, Tensor) and action.dim() == 1:
            t[TransitionKey.ACTION] = action.unsqueeze(0)

        # Complementary data
        comp = t.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp is not None and isinstance(comp, dict):
            new_comp = dict(comp)
            if "task" in new_comp and isinstance(new_comp["task"], str):
                new_comp["task"] = [new_comp["task"]]
            for k in ["index", "task_index"]:
                if k in new_comp and isinstance(new_comp[k], Tensor) and new_comp[k].dim() == 0:
                    new_comp[k] = new_comp[k].unsqueeze(0)
            t[TransitionKey.COMPLEMENTARY_DATA] = new_comp

        return t

    return step


class NormalizerStep:
    """Applies normalization or unnormalization to observations and actions."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        stats: dict[str, dict[str, Any]] | None = None,
        device: torch.device | str | None = None,
        eps: float = 1e-8,
        inverse: bool = False,
    ):
        self.features = features
        self.norm_map = norm_map
        self.stats = stats or {}
        self.device = device
        self.dtype = torch.float32
        self.eps = eps
        self.inverse = inverse
        self._tensor_stats: dict[str, dict[str, Tensor]] = _to_tensor(
            self.stats, device=self.device, dtype=self.dtype
        )

    def _to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = _to_tensor(self.stats, device=self.device, dtype=self.dtype)

    def _apply_transform(self, tensor: Tensor, key: str, feature_type: FeatureType) -> Tensor:
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        # Ensure stats match tensor device/dtype
        if key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self._to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]
        inv = self.inverse

        if norm_mode == NormalizationMode.MEAN_STD:
            mean, std = stats["mean"], stats["std"]
            denom = std + self.eps
            return tensor * std + mean if inv else (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX:
            mn, mx = stats["min"], stats["max"]
            denom = mx - mn
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom)
            if inv:
                return (tensor + 1) / 2 * denom + mn
            return 2 * (tensor - mn) / denom - 1

        if norm_mode == NormalizationMode.QUANTILES:
            q01, q99 = stats["q01"], stats["q99"]
            denom = q99 - q01
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom)
            if inv:
                return (tensor + 1.0) * denom / 2.0 + q01
            return 2.0 * (tensor - q01) / denom - 1.0

        if norm_mode == NormalizationMode.QUANTILE10:
            q10, q90 = stats["q10"], stats["q90"]
            denom = q90 - q10
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom)
            if inv:
                return (tensor + 1.0) * denom / 2.0 + q10
            return 2.0 * (tensor - q10) / denom - 1.0

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_t = transition.copy()

        # Observations
        obs = new_t.get(TransitionKey.OBSERVATION)
        if obs is not None:
            new_obs = dict(obs)
            for key, feature in self.features.items():
                if feature.type != FeatureType.ACTION and key in new_obs:
                    new_obs[key] = self._apply_transform(
                        torch.as_tensor(new_obs[key]), key, feature.type
                    )
            new_t[TransitionKey.OBSERVATION] = new_obs

        # Action
        action = new_t.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, Tensor):
            new_t[TransitionKey.ACTION] = self._apply_transform(action, ACTION, FeatureType.ACTION)

        return new_t


class TokenizerStep:
    """Tokenizes task text from complementary_data and adds tokens to observations."""

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 512,
        task_key: str = "task",
        padding_side: str = "right",
        padding: str = "max_length",
        truncation: bool = True,
    ):
        if not _transformers_available:
            raise ImportError("Install 'transformers' to use TokenizerStep.")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.task_key = task_key
        self.padding_side = padding_side
        self.padding = padding
        self.truncation = truncation

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_t = transition.copy()
        obs = new_t.get(TransitionKey.OBSERVATION)
        if obs is None:
            raise ValueError("Requires observation in transition")

        comp = new_t.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = comp.get(self.task_key)
        if task is None:
            raise ValueError("No task found in complementary data")
        if isinstance(task, str):
            task = [task]

        tokenized = self.tokenizer(
            task,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

        # Match device of existing tensors
        target_device = None
        for v in obs.values():
            if isinstance(v, Tensor):
                target_device = v.device
                break

        new_obs = dict(obs)
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"].to(dtype=torch.bool)
        if target_device is not None:
            ids = ids.to(target_device)
            mask = mask.to(target_device)
        new_obs[OBS_LANGUAGE_TOKENS] = ids
        new_obs[OBS_LANGUAGE_ATTENTION_MASK] = mask
        new_t[TransitionKey.OBSERVATION] = new_obs
        return new_t


class DeviceStep:
    """Moves all tensors in a transition to a target device."""

    def __init__(self, device: str = "cpu"):
        from lerobot.common.utils.utils import get_safe_torch_device

        self.tensor_device = get_safe_torch_device(device)
        self.non_blocking = "cuda" in str(self.tensor_device)

    def _process(self, tensor: Tensor) -> Tensor:
        # Multi-GPU: preserve tensor's GPU if both are on CUDA
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            target = tensor.device
        else:
            target = self.tensor_device
        if target.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.device != target:
            tensor = tensor.to(target, non_blocking=self.non_blocking)
        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_t = transition.copy()

        for key in [TransitionKey.ACTION, TransitionKey.REWARD, TransitionKey.DONE, TransitionKey.TRUNCATED]:
            v = new_t.get(key)
            if isinstance(v, Tensor):
                new_t[key] = self._process(v)

        for key in [TransitionKey.OBSERVATION, TransitionKey.COMPLEMENTARY_DATA]:
            d = new_t.get(key)
            if d is not None and isinstance(d, dict):
                new_t[key] = {
                    k: self._process(v) if isinstance(v, Tensor) else v for k, v in d.items()
                }

        return new_t


# ---------------------------------------------------------------------------
# PI05-specific step: prepare state + build prompt
# ---------------------------------------------------------------------------

def pi05_prepare_state_step(
    max_state_dim: int = 32, task_key: str = "task"
) -> Callable[[EnvTransition], EnvTransition]:
    """Pads state to max_state_dim, discretises it, and builds the full text prompt."""

    from lerobot.common.policies.pi05.modeling_pi05 import pad_vector

    def step(transition: EnvTransition) -> EnvTransition:
        t = transition.copy()

        state = t.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = t.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        state = pad_vector(deepcopy(state), max_state_dim)
        state_np = state.cpu().numpy()
        discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1

        prompts = []
        for i, task in enumerate(tasks):
            text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized[i]))
            prompts.append(f"Task: {text}, State: {state_str};\nAction: ")

        t[TransitionKey.COMPLEMENTARY_DATA][task_key] = prompts
        return t

    return step


# ---------------------------------------------------------------------------
# Public API: construct pre/post-processor pipelines for PI05
# ---------------------------------------------------------------------------

def make_pi05_pre_post_processors(
    config,  # PI05Config — not imported here to avoid circular import
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PI05ProcessorPipeline, PI05ProcessorPipeline]:
    """
    Build pre-processor and post-processor pipelines for PI05.

    Pre-processing:
      1. Rename observations (no-op by default)
      2. Add batch dimension
      3. Normalize observations & actions
      4. Prepare state & build prompt (discretise state, format task string)
      5. Tokenize text prompt (PaliGemma tokenizer)
      6. Move tensors to device

    Post-processing:
      1. Unnormalize actions
      2. Move tensors to CPU
    """
    input_steps = [
        rename_observations_step(rename_map={}),
        add_batch_dimension_step(),
        NormalizerStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            inverse=False,
        ),
        pi05_prepare_state_step(max_state_dim=config.max_state_dim),
        TokenizerStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceStep(device=config.device),
    ]

    output_steps = [
        NormalizerStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            inverse=True,
        ),
        DeviceStep(device="cpu"),
    ]

    return (
        PI05ProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_batch_to_transition,
            to_output=_transition_to_batch,
        ),
        PI05ProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=_policy_action_to_transition,
            to_output=_transition_to_policy_action,
        ),
    )
