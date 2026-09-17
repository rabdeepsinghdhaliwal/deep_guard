"""
Tests attribution.py against the REAL committed checkpoint
(models/deepguard_attribution.pth) -- no HTTP server, but real weight
loading, so this is slower than the pure unit tests.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from attribution import load_attribution_model, predict_generator
from model import GENERATOR_CLASSES

REAL_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "models" / "deepguard_attribution.pth"


@pytest.fixture(scope="module")
def attribution_state():
    state = load_attribution_model(REAL_CHECKPOINT_PATH, torch.device("cpu"))
    assert state is not None, f"Expected a real checkpoint at {REAL_CHECKPOINT_PATH}"
    return state


def test_checkpoint_class_order_matches_model_py(attribution_state):
    assert attribution_state["class_order"] == GENERATOR_CLASSES


def test_predict_generator_returns_valid_probability_distribution(attribution_state):
    random_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    result = predict_generator(attribution_state, random_image, torch.device("cpu"))

    assert result["predicted_generator"] in GENERATOR_CLASSES
    probs = result["probabilities"]
    assert set(probs.keys()) == set(GENERATOR_CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_predict_generator_names_the_highest_probability_class(attribution_state):
    random_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    result = predict_generator(attribution_state, random_image, torch.device("cpu"))
    top_by_value = max(result["probabilities"], key=result["probabilities"].get)
    assert result["predicted_generator"] == top_by_value


def test_missing_checkpoint_returns_none(tmp_path):
    fake_path = tmp_path / "does_not_exist.pth"
    assert load_attribution_model(fake_path, torch.device("cpu")) is None
