"""
Integration tests for the Content Registry endpoints. Uses the same
`client` fixture as everything else, whose registry files are
redirected to a session-scoped tmp directory by conftest.py -- these
tests never touch the real models/content_registry.* files.
"""
import pytest


def test_register_returns_signed_record(client, test_images):
    response = client.post(
        "/api/registry/register",
        files={"file": ("real3.jpg", test_images["real3.jpg"], "image/jpeg")},
        data={"title": "Test Registration — Real3"},
    )
    assert response.status_code == 200
    record = response.json()
    assert record["sha256_merkle_root"]
    assert record["signature"]
    assert record["title"] == "Test Registration — Real3"


def test_check_finds_the_exact_registered_image(client, test_images):
    client.post(
        "/api/registry/register",
        files={"file": ("real4.jpg", test_images["real4.jpg"], "image/jpeg")},
        data={"title": "Findable Entry"},
    )
    response = client.post(
        "/api/registry/check",
        files={"file": ("real4.jpg", test_images["real4.jpg"], "image/jpeg")},
    )
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert any(m["title"] == "Findable Entry" for m in matches)


def test_check_does_not_falsely_match_unrelated_image(client, test_images):
    client.post(
        "/api/registry/register",
        files={"file": ("ai1.png", test_images["ai1.png"], "image/png")},
        data={"title": "Unrelated To Ai2"},
    )
    response = client.post(
        "/api/registry/check",
        files={"file": ("ai2.png", test_images["ai2.png"], "image/png")},
    )
    matches = response.json()["matches"]
    assert not any(m["title"] == "Unrelated To Ai2" for m in matches)


def test_register_rejects_empty_title(client, test_images):
    response = client.post(
        "/api/registry/register",
        files={"file": ("real1.png", test_images["real1.png"], "image/png")},
        data={"title": "   "},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("payload_title", [
    "<img src=x onerror=alert(1)>",
    "'; DROP TABLE registry; --",
])
def test_register_accepts_hostile_titles_without_crashing(client, test_images, payload_title):
    # The server itself doesn't need to escape this -- escaping happens
    # client-side in registry.js (reviewed and fixed earlier this
    # session, a real stored-XSS finding). This test only pins that the
    # server-side path stays robust against hostile input, not that it
    # re-implements the same defence twice.
    response = client.post(
        "/api/registry/register",
        files={"file": ("real2.jpg", test_images["real2.jpg"], "image/jpeg")},
        data={"title": payload_title},
    )
    assert response.status_code == 200
    assert response.json()["title"] == payload_title


# ---------------------------------------------------------------------------
# Explanations: distinct features on upload, "why it matched" on check.
# ---------------------------------------------------------------------------

import base64
import io

from PIL import Image

import fingerprint
import local_features
import main


def _mona_lisa_crop_bytes(frac=0.6):
    image = Image.open(main.DEMO_ARTWORKS_DIR / "mona_lisa.jpg").convert("RGB")
    w, h = image.size
    cw, ch = int(w * frac), int(h * frac)
    crop = image.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_register_returns_distinct_features(client, test_images):
    response = client.post(
        "/api/registry/register",
        files={"file": ("real3.jpg", test_images["real3.jpg"], "image/jpeg")},
        data={"title": "Distinct Features Entry"},
    )
    features = response.json()["distinct_features"]
    assert features["points"]["count"] > 100
    assert len(features["areas"]) == 9
    assert sum(t["points"] for t in features["things"]) <= features["points"]["count"]
    assert features["palette"] and features["stored_kb"] > 0
    preview = Image.open(io.BytesIO(base64.b64decode(features["preview_jpeg_base64"])))
    assert preview.format == "JPEG"


def test_registration_record_still_verifies_once_distinct_features_is_set_aside(client, test_images):
    record = client.post(
        "/api/registry/register",
        files={"file": ("real1.png", test_images["real1.png"], "image/png")},
        data={"title": "Signature Still Valid"},
    ).json()
    record.pop("distinct_features")
    assert fingerprint.verify_signature(record["signer_public_key"].encode("ascii"), record, record["signature"])


def test_check_explains_a_crop_of_a_seeded_painting(client):
    response = client.post("/api/registry/check", files={"file": ("crop.jpg", _mona_lisa_crop_bytes(), "image/jpeg")})
    data = response.json()
    match = next(m for m in data["matches"] if m["title"] == "Mona Lisa")
    explanation = match["explanation"]
    assert explanation["available"] and explanation["confirmed"]
    assert explanation["shared_points"] >= 50
    assert abs(explanation["geometry"]["original_area_shown"] - 0.36) <= 0.03
    total = sum(t["share"] for t in explanation["common"]) + explanation["not_from_original_share"]
    assert abs(total - match["similarity"]) <= 1e-3
    assert data["upload_features"]["points"]["count"] > 0


def test_match_against_an_entry_without_features_is_reported_not_crashed(client, test_images):
    title = "Legacy Entry Without Features"
    client.post(
        "/api/registry/register",
        files={"file": ("real2.jpg", test_images["real2.jpg"], "image/jpeg")},
        data={"title": title},
    )
    registry = main.STATE["content_registry"]
    entry_id = registry.entry_ids_titled(title)[-1]
    registry._features_path(entry_id).unlink()  # as if registered before explanations existed

    response = client.post("/api/registry/check", files={"file": ("real2.jpg", test_images["real2.jpg"], "image/jpeg")})
    assert response.status_code == 200
    match = next(m for m in response.json()["matches"] if m["entry_id"] == entry_id)
    assert match["explanation"]["available"] is False


def test_registry_explains_without_the_namer(client, test_images, monkeypatch):
    monkeypatch.setitem(main.STATE, "concept_labeler", None)
    features = client.post(
        "/api/registry/register",
        files={"file": ("ai3.png", test_images["ai3.png"], "image/png")},
        data={"title": "No Namer Entry"},
    ).json()["distinct_features"]
    assert features["namer"] is None
    assert {t["name"] for t in features["things"]} == {f"{p} area" for p in local_features.AREA_POSITIONS}
