"""
Deep-Guard -- Generator Attribution (Phase 2, Idea 2).

Once /api/analyze has already decided an image is AI-generated, this
answers a second, harder question: which specific tool probably made it?
Every image generator leaves faint, characteristic statistical patterns
in its output -- the same underlying signal the frequency-domain panel
(frequency_analysis.py) surfaces visually, but learned end-to-end here
by a second, separately-trained classifier rather than measured by hand.

This is a SEPARATE model from Stage 1's real/fake detector (see
model.py's build_attribution_model() docstring for why), loaded only if
its checkpoint exists -- additive, like everything else in Phase 2. No
checkpoint means no generator_attribution field in the response, not a
broken app.

Honest limitation to have ready: GAN fingerprints often don't transfer
to diffusion models (the Modelship Attribution paper, cited in the
Phase 2 doc). This model was deliberately trained on BOTH families (see
GENERATOR_CLASSES in model.py) specifically so that gap is something
this project can measure and report on itself, not something to assume
away.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from model import GENERATOR_CLASSES, build_attribution_model

INPUT_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize((int(INPUT_SIZE * 256 / 224), int(INPUT_SIZE * 256 / 224))),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def load_attribution_model(checkpoint_path: Path, device: torch.device) -> Optional[dict]:
    """
    Returns {"model": ..., "class_order": [...]} or None if no checkpoint
    exists yet -- mirrors main.py's load_model() fallback-to-None pattern
    exactly, so a missing attribution checkpoint degrades the same way a
    missing Stage 1 checkpoint would: a clear log message, not a crash.
    """
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_order = checkpoint.get("class_order", GENERATOR_CLASSES)

    model = build_attribution_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return {"model": model, "class_order": class_order}


def predict_generator(attribution_state: dict, image: Image.Image, device: torch.device) -> dict:
    """
    Returns {"predicted_generator": str, "probabilities": {name: float, ...}},
    probabilities sorted highest first. Called only when the Stage 1
    verdict is already "manipulated" -- asking "which generator" about an
    image the detector itself thinks is real is a question this model
    was never trained to answer meaningfully.
    """
    model = attribution_state["model"]
    class_order = attribution_state["class_order"]

    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    paired = sorted(zip(class_order, probs), key=lambda p: p[1], reverse=True)
    return {
        "predicted_generator": paired[0][0],
        "probabilities": {name: round(float(p), 4) for name, p in paired},
    }
