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
# keys
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV = "observation.environment_state"
OBS_ENV_STATE = OBS_ENV
OBS_ROBOT = "observation.state"
OBS_STATE = OBS_ROBOT  # alias used by PI05 / upstream
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"
OBS_LANGUAGE_SUBTASK = OBS_STR + ".subtask"
OBS_LANGUAGE_SUBTASK_TOKENS = OBS_LANGUAGE_SUBTASK + ".tokens"
OBS_LANGUAGE_SUBTASK_ATTENTION_MASK = OBS_LANGUAGE_SUBTASK + ".attention_mask"
ACTION = "action"
ACTION_PREFIX = ACTION + "."
ACTION_TOKENS = ACTION + ".tokens"
ACTION_TOKEN_MASK = ACTION + ".token_mask"
REWARD = "next.reward"
TRUNCATED = "next.truncated"
DONE = "next.done"
INFO = "info"

ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

# openpi
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )
