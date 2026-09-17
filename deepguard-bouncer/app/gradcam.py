"""
Deep-Guard -- Grad-CAM explainability.

Turns the AI Bouncer's bare probability into a heatmap showing WHICH pixels
pushed the verdict toward "fake" (gradient-weighted class activation
mapping, applied to EfficientNet-B0's last convolutional block).

This only depends on the ARCHITECTURE (features[-1] -> pool -> one logit),
not on the learned weights, so it keeps working unchanged across model
retrains -- no code here needs to change when a new checkpoint is swapped
in via main.py.

Honesty note: the heatmap is computed on and overlaid onto the exact
224x224 square-resized-and-cropped image the model actually receives (see
main.py's `preprocess`), not the original full-resolution photo. That's a
deliberate choice -- it shows exactly what the model saw, rather than a
prettier image that would misrepresent where the model was actually
looking relative to its own input.
"""

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    input_tensor: a single preprocessed, NORMALIZED image tensor, shape
    (1, 3, H, W), on the model's device, requiring grad tracking (do not
    call this inside torch.no_grad()).

    Returns an (H, W) float32 array in [0, 1]: 1 = strongly pushed the
    verdict toward "fake", 0 = irrelevant to the decision.
    """
    activations: dict = {}
    gradients: dict = {}

    def forward_hook(_module, _inp, out):
        activations["value"] = out

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target_layer = model.features[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logit = model(input_tensor)  # shape (1, 1) -- the single fake-vs-real logit
        logit.sum().backward()

        acts = activations["value"][0]   # (C, h, w) feature-map activations
        grads = gradients["value"][0]    # (C, h, w) gradient of the logit w.r.t. those activations

        # Grad-CAM's core idea: how much would THIS logit change if THIS
        # channel's activation went up? Average that sensitivity spatially
        # per channel, then use it to weight how much each channel
        # contributes to the map.
        channel_weights = grads.mean(dim=(1, 2))
        cam = torch.einsum("c,chw->hw", channel_weights, acts)
        cam = F.relu(cam)  # only regions that push TOWARD fake, not away from it

        cam_min = cam.min()
        cam = cam - cam_min
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze(0).squeeze(0).detach().cpu().numpy()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        model.zero_grad(set_to_none=True)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """
    A dependency-free approximation of the classic blue-to-red 'jet'
    colormap (no matplotlib needed for three triangular functions).
    `values` in [0, 1] -> (H, W, 3) uint8 RGB.

    Public (not a module-private helper) because frequency_analysis.py
    reuses it too, so both explainability panels share one colour
    language instead of each rendering its own.
    """
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def render_heatmap_overlay(base_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> bytes:
    """
    Blends a heatmap (values in [0, 1], any resolution) over `base_image`
    (resized to match) and returns PNG bytes ready to embed as a data URI.
    """
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(base_image.size, Image.BILINEAR)
    heatmap_norm = np.asarray(heatmap_img).astype(np.float32) / 255.0
    colored = jet_colormap(heatmap_norm).astype(np.float32)

    base = np.asarray(base_image.convert("RGB")).astype(np.float32)
    blended = base * (1 - alpha) + colored * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format="PNG")
    return buffer.getvalue()
