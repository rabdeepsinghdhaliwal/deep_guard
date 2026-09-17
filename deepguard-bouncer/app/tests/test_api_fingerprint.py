"""
Integration tests for the fingerprint-related endpoints:
POST /api/fingerprint/compression-test and POST /api/fingerprint/verify.

test_oversized_verify_body_is_rejected pins the exact DoS-reopening
bug found and fixed this session: the first version of /verify used
Body(...), which has no size limit of its own, and accepted a 50MB
JSON body in ~0.65s -- reopening the same class of bug already closed
on every file-upload endpoint. If this ever regresses, this test
catches it immediately.
"""
import json

import pytest


@pytest.fixture()
def genuine_fingerprint(client, test_images):
    """A real, server-signed fingerprint record, obtained the same way
    the frontend gets one -- by actually analyzing a real image."""
    response = client.post(
        "/api/analyze",
        files={"file": ("real1.png", test_images["real1.png"], "image/png")},
    )
    return response.json()["fingerprint"]


def test_compression_test_changes_exact_hash_every_time(client, test_images):
    response = client.post(
        "/api/fingerprint/compression-test",
        files={"file": ("real1.png", test_images["real1.png"], "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    original_hash = data["original"]["sha256_merkle_root"]
    assert len(data["recompressed"]) > 0
    for row in data["recompressed"]:
        assert row["sha256_merkle_root"] != original_hash
        assert row["sha256_unchanged"] is False


def test_compression_test_keeps_perceptual_hash_drift_small(client, test_images):
    response = client.post(
        "/api/fingerprint/compression-test",
        files={"file": ("real1.png", test_images["real1.png"], "image/png")},
    )
    data = response.json()
    total_bits = data["phash_bits_total"]
    for row in data["recompressed"]:
        # "small" relative to the hash's total length -- recompression
        # should not move the visual fingerprint anywhere near as much
        # as it moves the exact one (which changes 100% of the time).
        assert row["phash_bits_different"] < total_bits * 0.15


def test_verify_genuine_record_is_valid(client, genuine_fingerprint):
    response = client.post("/api/fingerprint/verify", json=genuine_fingerprint)
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_verify_tampered_record_is_invalid(client, genuine_fingerprint):
    tampered = dict(genuine_fingerprint)
    original = tampered["sha256_merkle_root"]
    tampered["sha256_merkle_root"] = ("e" if original[0] == "f" else "f") + original[1:]

    response = client.post("/api/fingerprint/verify", json=tampered)
    assert response.status_code == 200
    assert response.json()["valid"] is False


def test_verify_record_missing_signature_is_rejected(client):
    response = client.post("/api/fingerprint/verify", json={"sha256_merkle_root": "abc"})
    assert response.status_code == 400


def test_oversized_verify_body_is_rejected(client):
    huge_record = {"junk": "x" * (2 * 1024 * 1024)}  # 2MB, over the 1MB limit
    response = client.post(
        "/api/fingerprint/verify",
        content=json.dumps(huge_record).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
