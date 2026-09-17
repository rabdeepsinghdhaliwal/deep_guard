"""
Deep-Guard -- frequency-domain explainability (Phase 2, Idea 5b).

Grad-CAM (gradcam.py) answers "where did the model look." This answers a
different question: "is there a measurable, physical reason to be
suspicious at all, independent of what the model thinks?"

Real cameras and AI generators leave different, measurable traces in an
image's frequency spectrum. A real photo's energy falls off smoothly from
low to high frequencies (natural optics + sensor noise). Many GANs leave
sharp, unnatural peaks at specific frequencies -- a quasi-periodic pattern
from the repeated upsampling layers inside the generator's architecture.
Diffusion models leave their own, different signature.

This is computed directly from the pixels via a Fourier transform -- no
model, no learned weights, no training data. It's a genuine physical
property of the image, which is exactly why it's worth showing alongside
Grad-CAM's model-derived heatmap rather than instead of it: one shows what
the model paid attention to, the other shows independent evidence a human
can verify with nothing but the pixels and a Fourier transform.

Honesty note, from actually testing this against the project's own 8
real/AI images rather than assuming the research transfers directly:
"peak detection" below is a simple heuristic (how far a pixel's energy
sits above the smooth radial trend expected at its frequency), and on
this small test set it did NOT separate real from AI -- max residual
z-scores clustered 2.6-3.6 for BOTH classes, with real photos trending
if anything slightly HIGHER than the AI ones. Plausible reasons: 8
images (4/4) proves nothing statistically; these images already carry
unknown prior compression/resizing that perturbs frequency content on
its own; 224x224 is much lower resolution than papers demonstrating this
effect typically use; and mixing PNG (lossless) with JPEG (whose own
8x8 DCT block structure adds its own periodic signal) confounds the
comparison further. This mirrors the Modelship Attribution paper's own
caveat (cited in the Phase 2 doc) that generator fingerprints often
don't transfer as cleanly as a first read of the literature suggests.

Conclusion: the spectrum IMAGE is real, correctly computed, independent
evidence worth showing on its own. The derived peak_count/anomaly_score
below should be shown as an exploratory number, never as a confident
claim of detection, until it's been validated against a properly sized,
controlled dataset (the Bias Map or Generator Attribution test sets,
once they exist) -- not asserted from a demo of 8 images.

SECOND FIX, found by actually running this live: the first version used
a hard PEAK_THRESHOLD_STD = 4.0 sigma cutoff to decide "is this a peak."
That's not just imperfect -- it was a real bug. The strongest deviation
measured across all 8 test images topped out around 3.5-3.6 sigma, so a
4.0 bar was mathematically guaranteed to never fire on anything, on any
image, ever. anomaly_score and peak_count read exactly 0.0 / 0 on every
single upload, which looks identical to "definitely nothing here" when
it actually means "this heuristic is structurally dead." Replaced the
hard threshold with a continuous score derived directly from the
strongest residual's z-score, calibrated (ANOMALY_Z_FLOOR/CEILING below)
against that same 2.6-3.6 range so real per-image variation is visible
-- still an unvalidated heuristic (see above), but now one that actually
moves instead of one that can't.

THIRD FIX, from round-1 adversarial testing: a solid grey square scored
anomaly_score=1.0 -- maximum "suspicious", on the least suspicious
possible image. See RAW_PIXEL_STD_FLOOR / has_negligible_texture()
below for the fix, and for the first (wrong) theory tried before it.
"""

import io

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift

from gradcam import jet_colormap

# anomaly_score is a linear rescaling of the single strongest residual's
# z-score (how many standard deviations it sits above the local noise
# floor) into [0, 1]. Calibrated against this project's own measurements
# across its 8 test images (max z-scores spanned ~2.6-3.6) so that range
# maps to a visibly varying ~0.2-0.5, rather than a fixed threshold that
# either always fires or never does. This is a display calibration, not
# a validated fake/real boundary -- see the module docstring.
ANOMALY_Z_FLOOR = 1.5    # z-score at/below this displays as 0.0
ANOMALY_Z_CEILING = 5.0  # z-score at/above this displays as 1.0

# A real photograph's grayscale pixels always vary by tens of units out
# of 0-255 (measured: 41-80 across this project's own test images). A
# perfectly flat colour swatch measures exactly 0.0. This threshold sits
# nowhere near either number, so it only ever fires on genuinely
# textureless input, never on a real photo.
#
# THIS REPLACED A FIRST ATTEMPT THAT DIDN'T WORK: the first fix checked
# the *post-normalization* residual std and short-circuited if it looked
# near-zero. Tested against an actual flat grey square, that check never
# fired -- its std_r measured ~0.017, inside the "normal" range, not
# near zero. Root cause, found by actually inspecting the numbers rather
# than assuming the first theory was right: compute_fft_spectrum's
# min-max normalization always stretches its output to fill exactly
# [0, 1], regardless of how much real energy the image had before that
# rescaling -- so a flat image's tiny, structurally-driven residual (an
# artifact of the Hann window not being radially symmetric, not of any
# real content) gets stretched to look just as significant as a real
# photo's actual structure. Checking texture in the RAW PIXELS, before
# any of that FFT/windowing/normalization machinery runs, sidesteps the
# whole problem instead of trying to out-think it downstream.
RAW_PIXEL_STD_FLOOR = 3.0


def has_negligible_texture(image: Image.Image) -> bool:
    """True for an image with essentially no real photographic content
    (a solid colour, or close to it) -- see RAW_PIXEL_STD_FLOOR above."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return bool(grayscale.std() < RAW_PIXEL_STD_FLOOR)

# Always report the top-K strongest residual locations, regardless of
# whether any absolute bar is crossed -- their z-scores speak for
# themselves, and a hard include/exclude cutoff is exactly the bug that
# made the old version go silent on every image.
MAX_PEAKS_REPORTED = 5


def compute_fft_spectrum(image: Image.Image) -> np.ndarray:
    """
    image: the same model-view (resized + centre-cropped, NOT normalized)
    image build_explainability() already computes for Grad-CAM's overlay --
    pass that same object in, so both panels show evidence about exactly
    what the model saw, not the original full-resolution upload.

    Returns a (H, W) float32 array of log-magnitude spectrum values,
    normalized to [0, 1] for display. Index [0, 0] of the raw FFT (the
    DC / average-brightness term) is centred via fftshift, so the plot's
    centre is "no change across the image" and its edges are the highest
    spatial frequencies.
    """
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)

    # A Hann window tapers the image to zero at its edges before the
    # transform. Without this, the hard edges of a cropped photo (a sharp
    # brightness discontinuity that isn't really "in" the image) leak
    # energy across every frequency and would swamp the real signal.
    h, w = grayscale.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = grayscale * window

    spectrum = fftshift(fft2(windowed))
    magnitude = np.abs(spectrum)
    log_magnitude = np.log1p(magnitude)

    min_val, max_val = log_magnitude.min(), log_magnitude.max()
    if max_val - min_val > 1e-8:
        log_magnitude = (log_magnitude - min_val) / (max_val - min_val)
    return log_magnitude.astype(np.float32)


def _radial_profile(spectrum: np.ndarray) -> np.ndarray:
    """
    Mean spectrum value at every integer pixel-distance from the centre
    (the DC term). This is "what a real photo's smooth falloff looks
    like, measured on THIS image" -- used as the expectation that actual
    per-pixel values get compared against, rather than a fixed universal
    curve, since absolute brightness/contrast varies image to image.
    """
    h, w = spectrum.shape
    cy, cx = h / 2, w / 2
    y, x = np.indices((h, w))
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    max_radius = radius.max()
    sums = np.bincount(radius.ravel(), spectrum.ravel(), minlength=max_radius + 1)
    counts = np.bincount(radius.ravel(), minlength=max_radius + 1)
    counts[counts == 0] = 1  # avoid divide-by-zero for any unreachable radius
    return sums / counts


def detect_spectral_peaks(spectrum: np.ndarray, source_image: Image.Image) -> dict:
    """
    Measures how far the spectrum's most extreme point sits above the
    smooth radial trend expected at its distance from the centre -- the
    signature GAN upsampling artifacts and some diffusion-model
    artifacts leave. Continuous, not a threshold pass/fail -- see the
    module docstring's "SECOND FIX" for why a hard cutoff was wrong.

    source_image: the same image compute_fft_spectrum(source_image) was
    given -- used only for the has_negligible_texture() guard (see
    RAW_PIXEL_STD_FLOOR above), never re-transformed.

    Returns:
        {
          "anomaly_score": float in [0, 1] -- how extreme the strongest
              deviation is, relative to this project's own observed range,
          "peak_count": int -- how many of the top-5 reported peaks have
              a z-score high enough to be worth a second look (z > 2.5,
              a labelling convenience, not a hard detection cutoff),
          "peaks": [{"radius_normalized": float, "magnitude": float,
              "z_score": float}, ...] -- always the top 5 strongest
              points, whatever their strength.
        }
    """
    if has_negligible_texture(source_image):
        return {"anomaly_score": 0.0, "peak_count": 0, "peaks": []}

    h, w = spectrum.shape
    cy, cx = h / 2, w / 2
    y, x = np.indices((h, w))
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius_int = radius.astype(np.int32)
    max_radius = radius_int.max()

    profile = _radial_profile(spectrum)
    expected = profile[np.clip(radius_int, 0, max_radius)]
    residual = spectrum - expected

    # Ignore the DC neighbourhood -- the average-brightness term and its
    # immediate surroundings are always the strongest thing in any
    # spectrum and carry no generator-artifact information.
    dc_radius = max(h, w) * 0.02
    outside_dc = radius >= dc_radius

    # mean/std for z-scoring must come from the SAME region peaks are
    # measured in -- the low-frequency band naturally carries far more
    # energy/variance than the high-frequency band an artifact would
    # show up in, and computing mean/std over the whole spectrum lets
    # that dominate (this was the first bug found here, before the
    # threshold-never-fires bug on top of it).
    mean_r, std_r = residual[outside_dc].mean(), residual[outside_dc].std()
    std_r = max(std_r, 1e-8)  # crash guard only -- has_negligible_texture() above is the real fix
    z_scores = np.where(outside_dc, (residual - mean_r) / std_r, -np.inf)

    normalizing_radius = max_radius if max_radius > 0 else 1
    order = np.argsort(z_scores.ravel())[::-1][:MAX_PEAKS_REPORTED]
    peak_coords = [np.unravel_index(i, z_scores.shape) for i in order]

    peaks = [
        {
            "radius_normalized": round(float(radius[coord]) / normalizing_radius, 4),
            "magnitude": round(float(spectrum[coord]), 4),
            "z_score": round(float(z_scores[coord]), 3),
        }
        for coord in peak_coords
    ]

    strongest_z = peaks[0]["z_score"] if peaks else 0.0
    anomaly_score = (strongest_z - ANOMALY_Z_FLOOR) / (ANOMALY_Z_CEILING - ANOMALY_Z_FLOOR)
    anomaly_score = min(1.0, max(0.0, anomaly_score))

    return {
        "anomaly_score": round(float(anomaly_score), 4),
        "peak_count": sum(1 for p in peaks if p["z_score"] > 2.5),
        "peaks": peaks,
    }


def render_spectrum_png(spectrum: np.ndarray) -> bytes:
    """Colours the spectrum with the same jet colormap Grad-CAM's heatmap
    uses, so the two panels read as one visual language. Unlike the
    heatmap, this is never blended over the photo -- a frequency
    spectrum has no meaningful spatial correspondence to paint it onto."""
    colored = jet_colormap(spectrum)
    buffer = io.BytesIO()
    Image.fromarray(colored).save(buffer, format="PNG")
    return buffer.getvalue()
