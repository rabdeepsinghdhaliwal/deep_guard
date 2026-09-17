"""
Unit tests for fingerprint.py -- no server, no models, pure functions.
"""
import fingerprint


def test_file_fingerprint_is_deterministic():
    data = b"the exact same bytes"
    a = fingerprint.compute_file_fingerprint(data)
    b = fingerprint.compute_file_fingerprint(data)
    assert a["sha256_merkle_root"] == b["sha256_merkle_root"]


def test_file_fingerprint_changes_with_one_byte():
    original = fingerprint.compute_file_fingerprint(b"deep-guard test payload")
    tampered = fingerprint.compute_file_fingerprint(b"Deep-guard test payload")  # one char flipped
    assert original["sha256_merkle_root"] != tampered["sha256_merkle_root"]


def test_file_fingerprint_reports_size_and_chunk_count():
    data = b"x" * 5000
    result = fingerprint.compute_file_fingerprint(data)
    assert result["file_size_bytes"] == 5000
    assert result["chunk_count"] == 1  # well under CHUNK_SIZE (1MB)


def test_hamming_distance_zero_for_identical_hashes():
    h = "a" * 64
    assert fingerprint.hamming_distance(h, h) == 0


def test_sign_and_verify_round_trip():
    private_pem, public_pem = fingerprint.generate_keypair()
    record = fingerprint.build_and_sign_record(
        private_pem, public_pem,
        sha256_merkle_root="abc123",
        phash="def456",
    )
    assert fingerprint.verify_signature(public_pem, record, record["signature"]) is True


def test_tampered_record_fails_verification():
    private_pem, public_pem = fingerprint.generate_keypair()
    record = fingerprint.build_and_sign_record(
        private_pem, public_pem,
        sha256_merkle_root="abc123",
        phash="def456",
    )
    tampered = dict(record)
    tampered["sha256_merkle_root"] = "TAMPERED"
    assert fingerprint.verify_signature(public_pem, tampered, record["signature"]) is False


def test_wrong_public_key_fails_verification():
    private_pem, _ = fingerprint.generate_keypair()
    _, other_public_pem = fingerprint.generate_keypair()  # unrelated keypair
    record = fingerprint.build_and_sign_record(private_pem, other_public_pem, sha256_merkle_root="abc123")
    assert fingerprint.verify_signature(other_public_pem, record, record["signature"]) is False
