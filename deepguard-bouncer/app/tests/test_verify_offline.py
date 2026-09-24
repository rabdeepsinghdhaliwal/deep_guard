"""
Tests for tools/verify_offline.py -- checking a signed record with nothing
but a public key.

The tool re-implements one rule from fingerprint.py on purpose (the exact
bytes that get signed), because standing alone is its whole point: it must
run with only `cryptography` installed and no Deep-Guard code. So these
tests pin it against the server's own signing code and against real server
answers saved to disk -- if the two ever drift apart, it fails here.

test_a_forger_who_embeds_their_own_key_fails pins the trust rule the tool
exists for: a record carries its signer's public key, so a record checked
against the key it carries proves nothing.
"""
import importlib.util
import json
from pathlib import Path

import pytest

import fingerprint

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "verify_offline.py"
_spec = importlib.util.spec_from_file_location("verify_offline", TOOL_PATH)
verify_offline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_offline)


@pytest.fixture(scope="module")
def keypair():
    return fingerprint.generate_keypair()  # (private_pem, public_pem)


@pytest.fixture()
def record(keypair):
    private_pem, public_pem = keypair
    return fingerprint.build_and_sign_record(
        private_pem, public_pem,
        **fingerprint.compute_file_fingerprint(b"a sealed file"),
        phash="0" * 64,
    )


def test_genuine_record_verifies_with_no_server(record, keypair):
    assert verify_offline.verify(record, keypair[1])


def test_one_changed_character_fails(record, keypair):
    root = record["sha256_merkle_root"]
    record["sha256_merkle_root"] = ("e" if root[0] == "f" else "f") + root[1:]
    assert not verify_offline.verify(record, keypair[1])


def test_a_forger_who_embeds_their_own_key_fails(keypair):
    forger_private, forger_public = fingerprint.generate_keypair()
    forged = fingerprint.build_and_sign_record(forger_private, forger_public, sha256_merkle_root="0" * 64)
    assert verify_offline.verify(forged, forger_public)   # passes against the key it carries...
    assert not verify_offline.verify(forged, keypair[1])  # ...fails against the key the user trusts


def test_saved_server_answers_verify_offline(client, test_images, tmp_path, capsys):
    """What a user actually does: save the server's JSON answers to disk and
    check them against the public key from the page footer (/api/health)."""
    key_file = tmp_path / "server_public_key.pem"
    key_file.write_text(client.get("/api/health").json()["signer_public_key"], encoding="ascii")
    upload = {"file": ("real1.png", test_images["real1.png"], "image/png")}

    answers = {
        "analysis": client.post("/api/analyze", files=upload),
        "registration": client.post("/api/registry/register", files=upload, data={"title": "Checked Offline"}),
        "check": client.post("/api/registry/check", files=upload),
    }
    assert answers["check"].json()["matches"], "the photo was just registered, so the check must match it"
    for name, answer in answers.items():
        saved = tmp_path / f"{name}.json"
        saved.write_bytes(answer.content)
        assert verify_offline.main([str(saved), str(key_file)]) == 0, name
    out = capsys.readouterr().out
    assert "INVALID" not in out
    assert "not signed: distinct_features" in out  # the registration's extra, named as unsigned


def test_exit_codes_say_what_happened(record, keypair, tmp_path):
    key_file = tmp_path / "key.pem"
    key_file.write_bytes(keypair[1])

    genuine = tmp_path / "genuine.json"
    genuine.write_text(json.dumps(record), encoding="utf-8-sig")  # with a BOM, as Windows PowerShell saves it
    assert verify_offline.main([str(genuine), str(key_file)]) == 0

    record["file_size_bytes"] += 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")
    assert verify_offline.main([str(tampered), str(key_file)]) == 1

    unsigned = tmp_path / "unsigned.json"
    unsigned.write_text(json.dumps({"sha256_merkle_root": "abc"}), encoding="utf-8")
    assert verify_offline.main([str(unsigned), str(key_file)]) == 2
