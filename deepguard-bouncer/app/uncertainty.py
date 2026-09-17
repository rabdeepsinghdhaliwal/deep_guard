"""
Deep-Guard -- confidence-awareness via Monte Carlo Dropout.

The classifier head keeps one Dropout layer specifically for this purpose
(see model.py). Normally at inference dropout is disabled (eval mode) so
every request gets one deterministic answer. Here, several forward passes
are run with JUST that dropout layer left active -- every other layer,
especially BatchNorm inside the backbone, stays in eval mode, since batch
statistics computed over copies of a single image would be meaningless.

Each pass randomly silences a different 20% of the classifier's input
features, so each pass is effectively a slightly different sub-network's
opinion. Close agreement across passes means the model is internally
consistent about this input; wide disagreement means the model itself
doesn't have a stable answer here. This needs no reference to the training
data (unlike a proper out-of-distribution detector built on training-set
feature statistics, which this project has no local copy of to build from).

Honest limitation, disclosed rather than hidden: this measures the model's
OWN uncertainty, not "does this resemble training data." A model that is
confidently and *consistently* wrong -- which is what the portrait/selfie
bug looks like empirically -- can show LOW variance here despite being
incorrect, because the mistake is systematic across sub-networks, not
borderline. See test_uncertainty_hypothesis.py for the empirical check of
exactly this, run against the known failure cases before this module was
wired into the app.
"""

import torch


def _set_dropout_training(model: torch.nn.Module, enabled: bool) -> None:
    """Toggles ONLY nn.Dropout submodules into train/eval mode, leaving
    BatchNorm and everything else untouched."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train(enabled)


def score_with_uncertainty(
    model: torch.nn.Module, input_tensor: torch.Tensor, n_samples: int = 16
) -> tuple[float, float]:
    """
    input_tensor: a single preprocessed image, shape (1, 3, H, W).
    Returns (mean_probability, std_probability) across n_samples
    dropout-perturbed forward passes, batched into one call for speed
    rather than looped.
    """
    _set_dropout_training(model, True)
    try:
        with torch.no_grad():
            batch = input_tensor.repeat(n_samples, 1, 1, 1)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs.mean().item(), probs.std().item()
    finally:
        _set_dropout_training(model, False)
