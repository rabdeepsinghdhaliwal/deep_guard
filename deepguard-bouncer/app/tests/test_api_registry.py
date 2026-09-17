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
