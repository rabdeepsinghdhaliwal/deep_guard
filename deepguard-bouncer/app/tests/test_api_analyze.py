"""
Integration tests for POST /api/analyze and GET /api/health, run
against the real app with real models loaded (via the session-scoped
`client` fixture in conftest.py). Mirrors the 91-check pipeline
regression run by hand earlier this session, now permanent.
"""
import pytest

from tests.conftest import TEST_IMAGE_FILES


def test_health_reports_model_loaded(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["signer_public_key"]


@pytest.mark.parametrize("filename", TEST_IMAGE_FILES)
def test_analyze_accepts_every_labeled_test_image(client, test_images, filename):
    content_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    response = client.post(
        "/api/analyze",
        files={"file": (filename, test_images[filename], content_type)},
    )
    assert response.status_code == 200
    data = response.json()

    assert 0.0 <= data["probability_fake"] <= 1.0

    # Fingerprint travels on every response, image or video.
    assert data["fingerprint"] is not None
    assert data["fingerprint"]["sha256_merkle_root"]
    assert data["fingerprint"]["signature"]

    # Explainability travels on every image response.
    assert data["explainability"] is not None
    assert data["explainability"]["heatmap_png_base64"]
    assert data["explainability"]["frequency_analysis"] is not None

    # generator_attribution is conditional on the SERVER's own verdict
    # gate (probability_fake >= 0.5) -- see main.py's build_generator_
    # attribution() call site. Not the client's stricter 80/20 headline
    # bucket, which is a separate, deliberate UI-only distinction.
    attribution = data.get("generator_attribution")
    if data["probability_fake"] >= 0.5:
        assert attribution is not None
        assert abs(sum(attribution["probabilities"].values()) - 1.0) < 1e-3
    else:
        assert attribution is None


def test_analyze_rejects_empty_file(client):
    response = client.post("/api/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_analyze_rejects_corrupt_image(client):
    response = client.post(
        "/api/analyze",
        files={"file": ("not_really_an_image.png", b"this is definitely not image data", "image/png")},
    )
    assert response.status_code == 422
