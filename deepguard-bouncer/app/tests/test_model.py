"""
Unit tests for model.py -- architecture shape checks only, no trained
weights loaded (pretrained=False, random init, fast).
"""
import torch

from model import build_model, build_attribution_model, GENERATOR_CLASSES, INPUT_SIZE


def test_build_model_outputs_single_logit():
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, 1)


def test_build_attribution_model_output_matches_generator_class_count():
    model = build_attribution_model(pretrained=False)
    model.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, len(GENERATOR_CLASSES))


def test_generator_classes_has_no_duplicates():
    # A silently duplicated entry would train fine and just misattribute
    # forever -- the same "silent" failure mode CLASS_ORDER's own
    # docstring warns about for the binary detector.
    assert len(GENERATOR_CLASSES) == len(set(GENERATOR_CLASSES))


def test_generator_classes_has_eight_entries():
    # Pins the specific number the attribution checkpoint was trained
    # against -- if this list's length ever changes without a retrain,
    # loading the existing checkpoint should fail loudly, not silently.
    assert len(GENERATOR_CLASSES) == 8
