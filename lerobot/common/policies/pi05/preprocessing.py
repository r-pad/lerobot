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
Normalization helpers for PI0.5 policy.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot.common.constants import ACTION, OBS_PREFIX
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def _to_tensor_stats(
    stats: dict[str, dict[str, Any]],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, dict[str, Tensor]]:
    """Convert a nested stats dict (may contain numpy/lists) to tensors."""
    result: dict[str, dict[str, Tensor]] = {}
    for key, sub in stats.items():
        result[key] = {}
        for stat_name, value in sub.items():
            if value is None:
                continue
            if isinstance(value, Tensor):
                t = value.to(dtype=dtype)
            else:
                t = torch.tensor(value, dtype=dtype)
            if device is not None:
                t = t.to(device=device)
            result[key][stat_name] = t
    return result


class Normalizer:
    """Applies normalization / unnormalization to batch dicts.

    Supports MEAN_STD, MIN_MAX, QUANTILES, QUANTILE10 modes per feature type.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType | str, NormalizationMode | str],
        stats: dict[str, dict[str, Any]] | None = None,
        eps: float = 1e-8,
    ):
        # Rebuild enum map if keys are strings (JSON round-trip)
        rebuilt_map: dict[FeatureType, NormalizationMode] = {}
        for k, v in norm_map.items():
            rebuilt_map[FeatureType(k) if isinstance(k, str) else k] = (
                NormalizationMode(v) if isinstance(v, str) else v
            )
        self.norm_map = rebuilt_map

        # Rebuild features if they came as dicts
        rebuilt_features: dict[str, PolicyFeature] = {}
        for key, val in features.items():
            if isinstance(val, dict):
                rebuilt_features[key] = PolicyFeature(type=FeatureType(val["type"]), shape=tuple(val["shape"]))
            else:
                rebuilt_features[key] = val
        self.features = rebuilt_features

        self.eps = eps
        self.stats = stats or {}
        self._tensor_stats: dict[str, dict[str, Tensor]] = _to_tensor_stats(self.stats)

    def _ensure_device_dtype(self, key: str, tensor: Tensor) -> None:
        """Move stats to match tensor device/dtype if needed."""
        if key not in self._tensor_stats:
            return
        first_stat = next(iter(self._tensor_stats[key].values()))
        if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
            self._tensor_stats = _to_tensor_stats(self.stats, device=tensor.device, dtype=tensor.dtype)

    def _apply_transform(self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool) -> Tensor:
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        self._ensure_device_dtype(key, tensor)
        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD:
            mean, std = stats["mean"], stats["std"]
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / (std + self.eps)

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom)
            if inverse:
                return (tensor + 1) / 2 * denom + min_val
            return 2 * (tensor - min_val) / denom - 1

        if norm_mode == NormalizationMode.QUANTILES:
            q01, q99 = stats["q01"], stats["q99"]
            denom = q99 - q01
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom)
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q01
            return 2.0 * (tensor - q01) / denom - 1.0

        if norm_mode == NormalizationMode.QUANTILE10:
            q10, q90 = stats["q10"], stats["q90"]
            denom = q90 - q10
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom)
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q10
            return 2.0 * (tensor - q10) / denom - 1.0

        raise ValueError(f"Unsupported normalization mode: {norm_mode}")

    def normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize observation and action keys in a batch dict (forward direction)."""
        for key, feature in self.features.items():
            if key in batch and isinstance(batch[key], Tensor):
                batch[key] = self._apply_transform(batch[key], key, feature.type, inverse=False)
        return batch

    def unnormalize_action(self, action: Tensor) -> Tensor:
        """Unnormalize a raw action tensor."""
        return self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=True)
