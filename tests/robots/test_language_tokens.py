from types import SimpleNamespace

import pytest
import torch

from lerobot.common.robot_devices.control_utils import maybe_add_language_tokens
from lerobot.common.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.common.utils.import_utils import _transformers_available


class _DummyPolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            type="pi05",
            max_state_dim=8,
            tokenizer_max_length=32,
        )


@pytest.mark.skipif(not _transformers_available, reason="transformers is required for language tokenization")
def test_maybe_add_language_tokens_populates_observation():
    observation = {
        "task": "Pick up the cube and place it in the bin.",
        OBS_STATE: torch.zeros(8),
    }

    maybe_add_language_tokens(observation, _DummyPolicy())

    tokens = observation[OBS_LANGUAGE_TOKENS]
    mask = observation[OBS_LANGUAGE_ATTENTION_MASK]

    assert tokens.ndim == 1
    assert mask.shape == tokens.shape
    assert mask.dtype == torch.bool


def test_maybe_add_language_tokens_requires_task_string():
    observation = {OBS_STATE: torch.zeros(8)}

    with pytest.raises(ValueError):
        maybe_add_language_tokens(observation, _DummyPolicy())

