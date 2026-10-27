"""The ledger's data structures on their own: hashing, Merkle proofs,
sealing, and every kind of tampering being caught where it happened."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import creators  # noqa: E402
import ledger_core as lc  # noqa: E402


@pytest.fixture
def keys():
    return lc.generate_keypair()


@pytest.fixture
def ledger(tmp_path, keys):
    return lc.Ledger(tmp_path / "data", *keys)


def rec(n: int) -> dict:
    return {"type": "registration", "record_id": f"r{n:03d}", "title": f"Photo {n}", "file_size_bytes": n}


def fill(ledger, sizes=(1, 3, 2)) -> None:
    n = 0
    for size in sizes:
        for _ in range(size):
            ledger.add_pending(rec(n))
            n += 1
        ledger.seal()


# ------------------------------------------------------------------ hashing

def test_canonical_bytes_are_sorted_compact_ascii():
    assert lc.canonical_bytes({"b": 1, "a": "é"}) == b'{"a":"\\u00e9","b":1}'


def test_floats_are_refused_anywhere_in_a_record():
    with pytest.raises(TypeError):
        lc.canonical_bytes({"reading": {"p": 0.5}})


@pytest.mark.parametrize("n", range(1, 10))
def test_every_merkle_proof_verifies_and_a_changed_record_does_not(n):
    records = [rec(i) for i in range(n)]
    root = lc.merkle_root_hex(records)
    for i in range(n):
        proof = lc.merkle_proof(records, i)
        assert lc.verify_merkle_proof(lc.record_leaf_hash(records[i]), proof, root)
        forged = {**records[i], "title": "forged"}
        assert not lc.verify_merkle_proof(lc.record_leaf_hash(forged), proof, root)


def test_odd_node_is_promoted_not_duplicated():
    # duplicating the last leaf would give [a,b,c] and [a,b,c,c] the same root
    a, b, c = rec(1), rec(2), rec(3)
    assert lc.merkle_root_hex([a, b, c]) != lc.merkle_root_hex([a, b, c, c])


def test_leaf_and_node_hashes_are_domain_separated():
    a, b = rec(1), rec(2)
    pair_root = lc.merkle_root_hex([a, b])
    assert pair_root != lc.record_leaf_hash(a)
    assert lc.merkle_root_hex([]) == lc.sha256_hex(b"")


# ------------------------------------------------------------------- ledger

def test_new_ledger_has_a_signed_genesis_block(ledger):
    blocks = ledger.read_blocks()
    assert len(blocks) == 1 and blocks[0]["index"] == 0
    assert blocks[0]["previous_hash"] == lc.GENESIS_PREVIOUS_HASH
    assert ledger.verify()["ok"]


def test_sealing_chains_blocks_and_empties_the_pending_pool(ledger):
    fill(ledger)
    blocks = ledger.read_blocks()
    assert [b["record_count"] for b in blocks] == [0, 1, 3, 2]
    for i in range(1, len(blocks)):
        assert blocks[i]["previous_hash"] == blocks[i - 1]["block_hash"]
    assert ledger.read_pending() == []
    assert ledger.seal() is None
    report = ledger.verify()
    assert report["ok"] and report["record_count"] == 6


def test_a_hand_edit_of_chain_jsonl_is_caught_at_that_block(ledger):
    fill(ledger)
    lines = ledger.chain_path.read_text(encoding="utf-8").splitlines()
    block = json.loads(lines[2])
    block["records"][1]["title"] = "Someone else's photo"
    lines[2] = json.dumps(block, sort_keys=True)
    ledger.chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = ledger.verify()
    assert not report["ok"] and report["first_bad_index"] == 2
    failed = [c["name"] for c in report["blocks"][2]["checks"] if not c["ok"]]
    assert failed == ["Records still match the Merkle root"]


def test_a_broken_json_line_is_reported_not_crashed_on(ledger):
    fill(ledger)
    lines = ledger.chain_path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1][:-5]
    ledger.chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = ledger.verify()
    assert report["blocks"][1]["checks"][0]["name"] == "Block can be read"
    assert 1 in report["bad_indexes"]


def test_edit_and_rehash_without_the_key_fails_the_signature_and_the_next_link(ledger, keys):
    fill(ledger)
    tampered, _ = lc.simulate_attack(ledger.read_blocks(), "edit_and_rehash", 1, keys[0])
    report = lc.verify_chain(tampered, keys[1])
    assert report["first_bad_index"] == 1
    assert [c["name"] for c in report["blocks"][1]["checks"] if not c["ok"]] == ["Signed by Deep-Guard's key"]
    assert [c["name"] for c in report["blocks"][2]["checks"] if not c["ok"]] == ["Points to the previous block"]


def test_deleting_a_block_is_caught(ledger, keys):
    fill(ledger)
    tampered, _ = lc.simulate_attack(ledger.read_blocks(), "delete_block", 2, keys[0])
    report = lc.verify_chain(tampered, keys[1])
    assert report["first_bad_index"] == 2


def test_insider_rewrite_passes_the_chain_but_not_a_witness(ledger, keys):
    fill(ledger)
    original = ledger.read_blocks()
    witness_notebook = [b["block_hash"] for b in original]
    tampered, _ = lc.simulate_attack(original, "insider_rewrite", 1, keys[0])
    assert lc.verify_chain(tampered, keys[1])["ok"]  # perfectly re-signed
    comparison = lc.compare_with_witness(tampered, witness_notebook)
    assert not comparison["agrees"] and comparison["first_mismatch"] == 1
    assert lc.compare_with_witness(original, witness_notebook)["agrees"]


def test_a_different_signing_key_is_refused(ledger):
    fill(ledger)
    _, other_public = lc.generate_keypair()
    report = lc.verify_chain(ledger.read_blocks(), other_public)
    assert not report["ok"] and report["first_bad_index"] == 0


def test_receipts_check_out_offline_and_a_forged_one_does_not(ledger, keys):
    fill(ledger)
    bundle = ledger.proof_bundle("r002")
    assert bundle["sealed"] and bundle["block_header"]["index"] == 2
    assert lc.verify_receipt(bundle, keys[1])["ok"]
    forged = json.loads(json.dumps(bundle))
    forged["record"]["title"] = "forged"
    assert not lc.verify_receipt(forged, keys[1])["ok"]


def test_status_collects_revocations_disputes_and_resolutions(ledger):
    ledger.add_pending(rec(1))
    ledger.add_pending({"type": "dispute", "record_id": "d1", "target_record_id": "r001", "reason": "mine"})
    ledger.add_pending({"type": "resolution", "record_id": "s1", "target_record_id": "r001", "outcome": "rejected"})
    ledger.add_pending({"type": "revocation", "record_id": "v1", "target_record_id": "r001", "reason": ""})
    ledger.seal()
    status = ledger.status_of("r001")
    assert status["revoked"]["record_id"] == "v1"
    assert [d["record_id"] for d in status["disputes"]] == ["d1"]
    assert [r["record_id"] for r in status["resolutions"]] == ["s1"]


def test_unknown_record_types_are_refused(ledger):
    with pytest.raises(ValueError):
        ledger.add_pending({"type": "payment", "record_id": "x"})


# ----------------------------------------------------------------- creators

def test_creator_signatures_verify_and_a_changed_record_fails(tmp_path, ledger):
    book = creators.CreatorBook(tmp_path)
    identity, created = book.get_or_create("Rabdeep")
    assert created and book.get_or_create("rabdeep")[1] is False
    signed = book.sign_record("Rabdeep", rec(7))
    assert lc.verify_creator_signature(signed) is True
    assert lc.verify_creator_signature({**signed, "title": "changed"}) is False
    ledger.add_pending(signed)
    ledger.seal()
    assert ledger.verify()["ok"]


def test_a_block_with_a_forged_creator_signature_fails(tmp_path, ledger):
    book = creators.CreatorBook(tmp_path)
    book.get_or_create("Arjun")
    signed = book.sign_record("Arjun", rec(8))
    signed["title"] = "changed after signing"
    ledger.add_pending(signed)
    ledger.seal()
    report = ledger.verify()
    assert [c["name"] for c in report["blocks"][1]["checks"] if not c["ok"]] == ["Every creator signature is valid"]


def test_even_deep_guard_cannot_edit_a_creator_signed_record(tmp_path, ledger, keys):
    """An insider with the real block key re-signs everything after editing a
    record -- the creator's own signature still gives it away."""
    book = creators.CreatorBook(tmp_path)
    book.get_or_create("Shubham")
    ledger.add_pending(book.sign_record("Shubham", rec(9)))
    ledger.seal()
    blocks = ledger.read_blocks()
    blocks[1]["records"][0]["title"] = "changed by an insider"
    blocks[1]["merkle_root"] = lc.merkle_root_hex(blocks[1]["records"])
    blocks[1]["block_hash"] = lc.block_hash_hex(blocks[1])
    blocks[1]["signature"] = lc.sign_block(keys[0], blocks[1]["block_hash"])
    report = lc.verify_chain(blocks, keys[1])
    assert [c["name"] for c in report["blocks"][1]["checks"] if not c["ok"]] == ["Every creator signature is valid"]


@pytest.mark.parametrize("bad", ["", "x", "a" * 41, "../etc", "<script>"])
def test_creator_names_are_validated(tmp_path, bad):
    with pytest.raises(ValueError):
        creators.CreatorBook(tmp_path).get_or_create(bad)
