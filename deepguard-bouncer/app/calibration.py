"""
Deep-Guard -- does "90%" mean right 9 times in 10?

A detector's percentage is not automatically a probability you can trust.
Calibration checks it the only way possible: take many labelled images,
group them by what the model said (60-70%, 70-80%, ...) and count how
often it was right in each group. A well-calibrated model is right about
75% of the time on the images it scores at 75%.

Temperature scaling (Guo et al., "On Calibration of Modern Neural
Networks", ICML 2017) is the standard one-number repair: divide the
model's raw score (its logit) by a temperature T before turning it into a
percentage. T > 1 softens an over-confident model, T < 1 sharpens a timid
one. It never moves an image across the 50% line, so it cannot change the
accuracy -- only how sure the number sounds.

What it cannot fix: a model that is confidently wrong on one kind of
image and confidently right on another. One T is shared by every image;
if a whole generator scores 0.1% "AI", no temperature turns that into a
warning. That failure needs better training data (roadmap Step 4), and
the exam set measures how often it happens.

This module is the maths only; evaluation/run_exam.py fits T on the
calibration set and measures the effect on the exam set, and
models/detector_calibration.json stores the result the app reads.
"""

import json
import math
from pathlib import Path
from typing import Optional, Sequence

from scipy.optimize import minimize_scalar

# Probabilities the model reports as exactly 0 or 1 carry no usable logit;
# clamp to this before taking one.
_EPS = 1e-7


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def apply_temperature(logit_value: float, temperature: float) -> float:
    """P(AI) after temperature scaling, from the model's raw logit."""
    return sigmoid(logit_value / temperature)


def negative_log_likelihood(logits: Sequence[float], labels: Sequence[int], temperature: float) -> float:
    """Mean NLL of the labels (1 = AI, 0 = real) under temperature T -- the
    quantity temperature scaling minimises."""
    total = 0.0
    for z, y in zip(logits, labels):
        p = min(max(apply_temperature(z, temperature), 1e-12), 1 - 1e-12)
        total -= math.log(p) if y else math.log(1.0 - p)
    return total / len(logits)


def fit_temperature(logits: Sequence[float], labels: Sequence[int]) -> float:
    """The T that minimises NLL, searched in log-space between 0.05 and 50."""
    if len(logits) != len(labels) or not logits:
        raise ValueError("need the same, non-zero number of logits and labels")
    if len(set(labels)) < 2:
        raise ValueError("need both real and AI examples to fit a temperature")
    result = minimize_scalar(lambda log_t: negative_log_likelihood(logits, labels, math.exp(log_t)),
                             bounds=(math.log(0.05), math.log(50.0)), method="bounded",
                             options={"xatol": 1e-4})
    return float(math.exp(result.x))


def reliability_table(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> list[dict]:
    """Groups images by predicted P(AI) into equal-width bins and reports,
    per bin, how many there are, what the model said on average, and how
    many really were AI. Empty bins are kept (count 0) so tables line up."""
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probabilities) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        n = len(idx)
        table.append({
            "bin_low": round(lo, 3),
            "bin_high": round(hi, 3),
            "count": n,
            "mean_predicted": round(sum(probabilities[i] for i in idx) / n, 4) if n else None,
            "fraction_ai": round(sum(labels[i] for i in idx) / n, 4) if n else None,
        })
    return table


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Average gap between "what the model said" and "how often it was so",
    weighted by how many images fall in each bin (0 = perfectly calibrated).
    Computed on P(AI), so a bin at 0.7 is compared with the fraction of AI
    images in it."""
    n = len(probabilities)
    ece = 0.0
    for row in reliability_table(probabilities, labels, bins):
        if row["count"]:
            ece += row["count"] / n * abs(row["mean_predicted"] - row["fraction_ai"])
    return ece


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared gap between P(AI) and the truth (0 or 1). Lower is better;
    always guessing 50% scores 0.25."""
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def load_calibration(path: Path) -> Optional[dict]:
    """Reads models/detector_calibration.json if it exists -- additive, like
    every optional model file in this project: no file, no calibration."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    t = data.get("detector_temperature")
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not 0.05 <= t <= 50:
        raise ValueError(f"{path}: detector_temperature must be a number between 0.05 and 50, got {t!r}")
    if "attribution_unknown_threshold" in data:
        u = data["attribution_unknown_threshold"]
        # A malformed value would otherwise surface only as a TypeError inside
        # the source panel, silently hiding the whole panel.
        if not isinstance(u, (int, float)) or isinstance(u, bool) or not 0.0 <= u <= 1.0:
            raise ValueError(f"{path}: attribution_unknown_threshold must be a number between 0 and 1, got {u!r}")
    return data
