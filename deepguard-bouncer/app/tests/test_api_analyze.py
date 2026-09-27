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


def test_calibrated_reading_travels_alongside_the_raw_one(client, test_images):
    """Step 0: with models/detector_calibration.json present, every image
    answer also carries the temperature-scaled reading -- same side of 50%
    as the raw one (temperature scaling can't cross it), and the verdict
    itself still comes from the raw number. Without the file, no field."""
    import main
    data = client.post("/api/analyze",
                       files={"file": ("real3.jpg", test_images["real3.jpg"], "image/jpeg")}).json()
    if main.STATE["calibration"] is None:
        assert "calibrated_probability_fake" not in data
        return
    assert 0.0 <= data["calibrated_probability_fake"] <= 1.0
    assert (data["calibrated_probability_fake"] >= 0.5) == (data["probability_fake"] >= 0.5)
    assert data["calibration_temperature"] == main.STATE["calibration"]["detector_temperature"]
    assert data["verdict"] == ("manipulated" if data["probability_fake"] >= 0.5 else "authentic")


def test_source_panel_answers_known_or_unknown_when_calibrated(client):
    """The shipped calibration file includes the "unknown" threshold, so the
    source panel's answer says whether it recognises the tool at all."""
    import main
    from tests.conftest import TEST_IMAGES_DIR
    if main.STATE["calibration"] is None or main.STATE["attribution"] is None:
        return
    # A DALL-E 3 image from the canary set: the detector flags it (100%),
    # and DALL-E 3 is not one of the panel's 8 tools.
    path = TEST_IMAGES_DIR.parent / "generalization_test" / "held_out_images" / "dalle3" / "dalle3_000.webp"
    data = client.post("/api/analyze", files={"file": (path.name, path.read_bytes(), "image/webp")}).json()
    attr = data["generator_attribution"]
    assert attr["answer"] in ("known", "unknown")
    top = next(iter(attr["probabilities"].values()))  # sorted highest first, rounded to 4 places
    if abs(top - attr["unknown_threshold"]) > 1e-4:
        assert attr["answer"] == ("known" if top >= attr["unknown_threshold"] else "unknown")


def test_analyze_rejects_empty_file(client):
    response = client.post("/api/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_analyze_rejects_corrupt_image(client):
    response = client.post(
        "/api/analyze",
        files={"file": ("not_really_an_image.png", b"this is definitely not image data", "image/png")},
    )
    assert response.status_code == 422
