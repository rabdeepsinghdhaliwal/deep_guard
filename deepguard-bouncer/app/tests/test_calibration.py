"""
Tests calibration.py -- pure maths, no model weights, so these are fast.
The key properties: temperature scaling never moves an image across 50%,
the fit recovers a known temperature, and the error measures are 0 for a
perfectly honest model and large for a lying one.
"""
import json
import math
import random

import pytest

from calibration import (apply_temperature, brier_score, expected_calibration_error, fit_temperature,
                         load_calibration, logit, reliability_table, sigmoid)


def test_logit_and_sigmoid_are_inverses():
    for p in (0.001, 0.2, 0.5, 0.8, 0.999):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)


def test_logit_survives_exact_zero_and_one():
    # The detector really does report 0.0 and 1.0 (rounded); neither may crash.
    assert math.isfinite(logit(0.0)) and logit(0.0) < -10
    assert math.isfinite(logit(1.0)) and logit(1.0) > 10


def test_temperature_never_changes_which_side_of_fifty_percent():
    for z in (-6.0, -0.3, 0.0, 0.4, 7.5):
        for t in (0.3, 1.0, 2.5, 10.0):
            assert (apply_temperature(z, t) >= 0.5) == (sigmoid(z) >= 0.5)


def test_temperature_above_one_softens_below_one_sharpens():
    assert apply_temperature(4.0, 2.0) < sigmoid(4.0)
    assert apply_temperature(4.0, 0.5) > sigmoid(4.0)
    assert apply_temperature(4.0, 1.0) == pytest.approx(sigmoid(4.0))


def test_fit_recovers_a_known_temperature():
    # Draw labels from a model whose honest logits are true_logit / 3: the
    # raw logits are 3x over-confident, so the fit should land near T = 3.
    rng = random.Random(0)
    logits, labels = [], []
    for _ in range(4000):
        z = rng.uniform(-12, 12)
        logits.append(z)
        labels.append(int(rng.random() < sigmoid(z / 3.0)))
    assert fit_temperature(logits, labels) == pytest.approx(3.0, rel=0.15)


def test_fit_refuses_one_class_data():
    with pytest.raises(ValueError):
        fit_temperature([1.0, 2.0, 3.0], [1, 1, 1])


def test_reliability_table_counts_every_image_once():
    probs = [0.0, 0.05, 0.15, 0.5, 0.95, 1.0]
    table = reliability_table(probs, [0, 0, 1, 1, 1, 1], bins=10)
    assert len(table) == 10
    assert sum(b["count"] for b in table) == len(probs)
    assert table[-1]["count"] == 2  # 0.95 and exactly 1.0 both land in the top bin


def test_ece_is_zero_for_an_honest_model_and_large_for_a_liar():
    honest = expected_calibration_error([0.25] * 4 + [0.75] * 4, [1, 0, 0, 0, 1, 1, 1, 0])
    liar = expected_calibration_error([0.99] * 4 + [0.01] * 4, [0, 0, 0, 0, 1, 1, 1, 1])
    assert honest == pytest.approx(0.0, abs=1e-9)
    assert liar > 0.9


def test_brier_score_reference_points():
    assert brier_score([0.5, 0.5], [0, 1]) == pytest.approx(0.25)
    assert brier_score([0.0, 1.0], [0, 1]) == pytest.approx(0.0)


def test_load_calibration_is_optional_and_validated(tmp_path):
    assert load_calibration(tmp_path / "missing.json") is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"detector_temperature": 2.1}))
    assert load_calibration(good)["detector_temperature"] == 2.1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"detector_temperature": -1}))
    with pytest.raises(ValueError):
        load_calibration(bad)


def test_load_calibration_rejects_a_malformed_unknown_threshold(tmp_path):
    for bad in ("0.99", 1.5, -0.1, True):
        f = tmp_path / "bad_threshold.json"
        f.write_text(json.dumps({"detector_temperature": 2.0, "attribution_unknown_threshold": bad}))
        with pytest.raises(ValueError, match="attribution_unknown_threshold"):
            load_calibration(f)
