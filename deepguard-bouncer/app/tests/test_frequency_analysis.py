"""
Unit tests for frequency_analysis.py -- no server, no ML model.

test_has_negligible_texture_true_for_flat_image and
test_detect_spectral_peaks_on_flat_image_is_zero pin the exact bug
found and fixed this session: a flat/near-solid-colour image used to
report a non-zero, misleading anomaly score because the FFT's own
min-max normalization stretched a tiny structural artifact (from the
Hann window) to fill [0, 1] as if it were real content. If either of
these ever starts failing again, that regression is back.
"""
from pathlib import Path

import pytest
from PIL import Image

import frequency_analysis as freq

TEST_IMAGES_DIR = Path(__file__).resolve().parents[3] / "test_images"


@pytest.fixture()
def flat_image():
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture()
def real_photo():
    return Image.open(TEST_IMAGES_DIR / "real1.png").convert("RGB").resize((224, 224))


def test_has_negligible_texture_true_for_flat_image(flat_image):
    assert freq.has_negligible_texture(flat_image) is True


def test_has_negligible_texture_false_for_real_photo(real_photo):
    assert freq.has_negligible_texture(real_photo) is False


def test_detect_spectral_peaks_on_flat_image_is_zero(flat_image):
    spectrum = freq.compute_fft_spectrum(flat_image)
    result = freq.detect_spectral_peaks(spectrum, flat_image)
    assert result["anomaly_score"] == 0.0
    assert result["peak_count"] == 0
    assert result["peaks"] == []


def test_detect_spectral_peaks_on_real_photo_has_expected_shape(real_photo):
    spectrum = freq.compute_fft_spectrum(real_photo)
    result = freq.detect_spectral_peaks(spectrum, real_photo)
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert isinstance(result["peak_count"], int)
    assert len(result["peaks"]) <= 5
    for peak in result["peaks"]:
        assert {"radius_normalized", "magnitude", "z_score"} <= peak.keys()


def test_compute_fft_spectrum_output_is_normalized(real_photo):
    spectrum = freq.compute_fft_spectrum(real_photo)
    assert spectrum.shape == (224, 224)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0
