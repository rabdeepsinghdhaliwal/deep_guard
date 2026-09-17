"""
Deep-Guard Stage 1 ("The Bouncer") — model architecture.

CRITICAL: This exact function is duplicated, line for line, inside the
Colab training notebook (colab_notebook/DeepGuard_Bouncer_Training.ipynb,
the cell titled "Step 4 — model architecture"). The two environments do
not share a Python import path — Colab runs in the cloud, this file runs
on your laptop — so the architecture has to be declared twice. If you
ever change this function, you MUST change the matching cell in the
notebook and retrain, or the saved .pth weights will fail to load, or
worse, will load into the wrong layer shapes silently.

Design choice: a single-logit binary head, not a 2-class softmax. This
means the model outputs ONE number (a logit) which we pass through a
sigmoid to get a single probability P(fake) in [0, 1]. This is simpler
to reason about end-to-end than a 2-class softmax and maps directly onto
the UI's "confidence" readout.
"""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Fixed label convention used everywhere in this project. This is NOT the
# order torchvision.datasets.ImageFolder assigns by default (which sorts
# class folder names alphabetically — "fake" would come before "real").
# We deliberately override that in the notebook's dataset loader so that
# index 0 always means "real" and index 1 always means "fake", matching
# the sigmoid output convention below. Getting this backwards is the
# single most common — and most silent — bug in this kind of pipeline:
# the model trains, the accuracy numbers look fine, and every verdict is
# simply inverted. Treat this list as the one source of truth.
CLASS_ORDER = ["real", "fake"]

# Standard EfficientNet-B0 preprocessing. Hardcoded here (rather than
# pulled from EfficientNet_B0_Weights.DEFAULT.transforms()) so local
# inference never depends on torchvision's pretrained-weights metadata
# being reachable. The notebook uses the same four numbers for training.
INPUT_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = False) -> nn.Module:
    """
    Builds the EfficientNet-B0 backbone used for Deep-Guard Stage 1.

    Args:
        pretrained: if True, loads ImageNet weights (used only in the
            Colab notebook, at the start of training). The local
            FastAPI backend always calls this with pretrained=False,
            because it immediately overwrites every weight with the
            fine-tuned checkpoint anyway — downloading ImageNet weights
            first would be pure wasted bandwidth and time.

    Returns:
        An nn.Module whose forward() returns a single logit per image,
        shape (batch, 1). Apply torch.sigmoid() to get P(fake).
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # EfficientNet-B0's stock classifier head is Dropout -> Linear(1280, 1000)
    # (1000 ImageNet classes). Replace only the final Linear layer with a
    # single-output layer; keep the Dropout, since it's still useful
    # regularization during our fine-tuning.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model


# ---------------------------------------------------------------------------
# Phase 2, Idea 2 -- Generator Attribution (a SEPARATE model, not a second
# head on the one above -- see the Phase 2 build plan's Decision 1: a
# shared multi-task model risks the two tasks fighting each other during
# training, and a second model is simpler to build, debug, and roll back
# independently of Stage 1, which stays completely untouched by any of
# this).
# ---------------------------------------------------------------------------

# 8 classes, chosen from ArtiFact's 25 for genuine architectural diversity
# on both sides, not just a coarse GAN-vs-diffusion split -- so a
# generalization test (train on some, hold out others within each family)
# is actually meaningful. Order matters: index position IS the class
# label everywhere this list is used, in this file, the training
# notebook, and attribution.py -- same one-source-of-truth discipline as
# CLASS_ORDER above, for the same reason (a silently reordered list here
# would retrain nothing and just relabel every prediction).
GENERATOR_CLASSES = [
    # GAN family
    "stylegan2",       # classic unconditional GAN
    "pro_gan",         # progressive growing
    "big_gan",         # class-conditional
    "cycle_gan",       # image-to-image translation -- architecturally distinct from the other three
    # Diffusion family
    "ddpm",            # foundational / vanilla diffusion, no conditioning
    "latent_diffusion",  # Stable Diffusion's direct architectural ancestor
    "stable_diffusion",  # the most widely recognized name in this list
    "glide",           # text-guided, a different diffusion lineage (OpenAI)
]


def build_attribution_model(pretrained: bool = False) -> nn.Module:
    """
    Same EfficientNet-B0 backbone as build_model(), a different head:
    Linear(in_features, len(GENERATOR_CLASSES)) instead of Linear(in_features, 1).
    Multi-class, not binary -- softmax over GENERATOR_CLASSES, not a
    single sigmoid probability. Train with nn.CrossEntropyLoss (expects
    raw logits, applies log-softmax internally) and take
    torch.softmax(logits, dim=-1) at inference for a probability per
    generator.

    CRITICAL, same warning as build_model() above: this must match the
    matching cell in DeepGuard_Attribution_Training.ipynb exactly. If
    GENERATOR_CLASSES' length or order ever changes here, the notebook's
    copy must change identically and the model must be retrained -- a
    saved checkpoint's output layer shape and meaning are both frozen at
    training time.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(GENERATOR_CLASSES))

    return model
