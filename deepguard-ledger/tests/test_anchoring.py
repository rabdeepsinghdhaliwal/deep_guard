"""Anchoring to Bitcoin: checked offline. The real network (calendars, block
explorers) is replaced by fakes everywhere except the official example proof,
whose Bitcoin block is pinned from two explorers (see fixtures/ots/README.md)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from opentimestamps.core.notary import PendingAttestation
from opentimestamps.core.op import OpSHA256
from opentimestamps.core.timestamp import Timestamp

LEDGER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEDGER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchoring  # noqa: E402
import ledger_core as lc  # noqa: E402
from fake_bitcoin import FakeBitcoin  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ots"
EXAMPLE_BLOCK = {"height": 358391, "block_hash": "000000000000000003e892881a8cdcdc117c06d444057c98b6f04a9ee75a2319",
                 "merkle_root": "8a1b66ecb7cbd07d8139a7e7d7f2c41aab1f5009b8364aaf61d03ad245e47e00",
                 "time": 1432827678, "time_utc": "2015-05-28T15:41:18+00:00",
                 "sources": ["pinned"], "single_source": False}


REAL_ANCHOR = Path(__file__).resolve().parent / "fixtures" / "anchor"
REAL_BLOCKS = {  # pinned from blockstream.info and blockcypher.com, see fixtures/anchor/README.md
    974545: {"height": 974545, "block_hash": "00000000000000000000076e8d490afd1599c14cfb7bf3d6c3f7c520731653d8",
             "merkle_root": "01c66ef9ea807cff37f391608735a4b8fcd0c2239c3796c1cb3c37ec15413de0", "time": 1793809304,
             "time_utc": "2026-11-04T16:21:44+00:00", "sources": ["pinned"], "single_source": False},
    974547: {"height": 974547, "block_hash": "00000000000000000001ca259230f213d5f48020bdbd67c13c70f5c6804a5662",
             "merkle_root": "903e6c40673c32c7921d7ea5a899dca0c852a9efa873c62def0e0611fac3062d", "time": 1793809724,
             "time_utc": "2026-11-04T16:28:44+00:00", "sources": ["pinned"], "single_source": False},
}


def pinned(height):
    assert height == EXAMPLE_BLOCK["height"]
    return EXAMPLE_BLOCK


# ------------------------------------------------- the official example proof

def test_the_official_example_proof_reaches_its_bitcoin_block():
    header = (FIXTURES / "hello-world.txt").read_bytes()
    proof = (FIXTURES / "hello-world.txt.ots").read_bytes()
    result = anchoring.verify_files(header, proof, lookup=pinned)
    assert result["ok"] and result["status"] == anchoring.STATUS_CONFIRMED, result["checks"]
    assert result["bitcoin"]["height"] == 358391 and result["bitcoin"]["time"] == 1432827678
    bitcoin_paths = [p for p in result["paths"] if p["attestation"]["type"] == "bitcoin"]
    assert bitcoin_paths and bitcoin_paths[0]["msg"] == bytes.fromhex(EXAMPLE_BLOCK["merkle_root"])[::-1].hex()


def test_replaying_the_exported_path_gives_the_merkle_root():
    """The explorer page replays these steps in the browser; do it here the
    same way, by hand, to pin what the page relies on."""
    header = (FIXTURES / "hello-world.txt").read_bytes()
    result = anchoring.verify_files(header, (FIXTURES / "hello-world.txt.ots").read_bytes(), lookup=pinned)
    path = next(p for p in result["paths"] if p["attestation"]["type"] == "bitcoin")
    msg = hashlib.sha256(header).digest()
    for step in path["ops"]:
        if step["op"] == "append":
            msg = msg + bytes.fromhex(step["arg"])
        elif step["op"] == "prepend":
            msg = bytes.fromhex(step["arg"]) + msg
        elif step["op"] == "sha256":
            msg = hashlib.sha256(msg).digest()
        elif step["op"] == "ripemd160":
            msg = hashlib.new("ripemd160", msg).digest() if "ripemd160" in hashlib.algorithms_available else None
            if msg is None:
                pytest.skip("this Python has no RIPEMD-160")
        else:
            pytest.fail(f"unexpected step {step}")
    assert msg.hex() == path["msg"]
    assert anchoring.merkle_root_matches(msg, EXAMPLE_BLOCK["merkle_root"])


def test_one_changed_byte_breaks_the_example_proof():
    header = bytearray((FIXTURES / "hello-world.txt").read_bytes())
    header[0] ^= 1
    result = anchoring.verify_files(bytes(header), (FIXTURES / "hello-world.txt.ots").read_bytes(), lookup=pinned)
    assert not result["ok"] and result["status"] == anchoring.STATUS_FAILED
    assert any(c["name"] == "Proof is for this header" and c["ok"] is False for c in result["checks"])


def test_a_wrong_merkle_root_is_caught():
    wrong = {**EXAMPLE_BLOCK, "merkle_root": "00" * 32}
    result = anchoring.verify_files((FIXTURES / "hello-world.txt").read_bytes(),
                                    (FIXTURES / "hello-world.txt.ots").read_bytes(), lookup=lambda h: wrong)
    assert not result["ok"] and result["status"] == anchoring.STATUS_FAILED


def test_the_offline_check_stops_before_bitcoin():
    result = anchoring.verify_files((FIXTURES / "hello-world.txt").read_bytes(),
                                    (FIXTURES / "hello-world.txt.ots").read_bytes(), lookup=None)
    assert result["ok"] and result["status"] == anchoring.STATUS_CLAIMED
    assert any("not looked up" in c["detail"] for c in result["checks"])


def test_a_damaged_proof_file_is_reported_not_crashed_on():
    result = anchoring.verify_files(b"header", b"not an ots file", lookup=pinned)
    assert not result["ok"] and result["checks"][-1]["name"] == "Proof file can be read"


# ------------------------------------------------- a real Deep-Guard anchor

def test_a_real_deep_guard_anchor_reaches_two_bitcoin_blocks():
    """Block #2 of a development ledger, anchored on 2026-11-04 and put into
    Bitcoin by two different calendars (blocks 974545 and 974547)."""
    header = (REAL_ANCHOR / "block-000002.header.json").read_bytes()
    proof = (REAL_ANCHOR / "block-000002.header.json.ots").read_bytes()
    assert json.loads(header)["index"] == 2
    block_hash = hashlib.sha256(header).hexdigest()
    result = anchoring.verify_files(header, proof, anchored_hash=block_hash, chain_hash=block_hash,
                                    lookup=REAL_BLOCKS.__getitem__)
    assert result["ok"] and result["status"] == anchoring.STATUS_CONFIRMED, result["checks"]
    assert result["bitcoin"]["height"] == 974545  # the earliest block proves the earliest date
    reached = [c for c in result["checks"] if c["name"].startswith("Proof reaches Bitcoin block")]
    assert len(reached) == 2 and all(c["ok"] for c in reached)


def test_the_real_anchor_fails_against_another_block():
    header = (REAL_ANCHOR / "block-000002.header.json").read_bytes()
    proof = (REAL_ANCHOR / "block-000002.header.json.ots").read_bytes()
    swapped = {974545: REAL_BLOCKS[974547], 974547: REAL_BLOCKS[974545]}
    result = anchoring.verify_files(header, proof, lookup=swapped.__getitem__)
    assert not result["ok"] and result["status"] == anchoring.STATUS_FAILED


# ------------------------------------------------- explorers have to agree

def test_explorers_must_agree():
    a = ("a", lambda h: {"block_hash": "11", "merkle_root": "22", "time": 1})
    b = ("b", lambda h: {"block_hash": "11", "merkle_root": "33", "time": 1})
    with pytest.raises(anchoring.AnchorError, match="disagree"):
        anchoring.bitcoin_block(5, explorers=(a, b))


def test_one_explorer_down_is_tolerated_but_flagged():
    def down(h):
        raise OSError("unreachable")
    a = ("a", lambda h: {"block_hash": "11", "merkle_root": "22", "time": 1})
    block = anchoring.bitcoin_block(5, explorers=(("down", down), a))
    assert block["sources"] == ["a"] and block["single_source"]
    with pytest.raises(anchoring.AnchorError, match="online"):
        anchoring.bitcoin_block(5, explorers=(("down", down),))


# ------------------------------------------------- the whole life of an anchor

@pytest.fixture
def ledger(tmp_path):
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
    public_pem = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    ledger = lc.Ledger(tmp_path / "data", private_pem, public_pem)
    ledger.add_pending({"type": "resolution", "record_id": "r1", "target_record_id": "x", "dispute_record_id": "y",
                        "outcome": "rejected", "note": "", "resolved_by": "test", "filed_at": lc.now_iso()})
    ledger.seal()
    return ledger


def test_the_header_file_hashes_to_the_block_hash(ledger):
    for block in ledger.read_blocks():
        assert hashlib.sha256(anchoring.header_bytes(block)).hexdigest() == block["block_hash"]


def test_pending_then_confirmed(ledger, tmp_path):
    fake = FakeBitcoin()
    book = anchoring.AnchorBook(tmp_path / "data")
    block = ledger.read_blocks()[-1]
    entry = book.create(block, calendars=("https://a.example", "https://b.example"), submit=fake.submit)
    assert entry["status"] == anchoring.STATUS_PENDING and entry["block_index"] == 1
    assert all(c["ok"] for c in entry["calendars"]) and len(fake.submitted) == 2
    # the calendars got a salted hash, never the block hash itself
    assert all(digest != bytes.fromhex(block["block_hash"]) for _, digest in fake.submitted)
    header, proof = book.files(entry)
    assert hashlib.sha256(header).hexdigest() == block["block_hash"]
    assert anchoring.verify_files(header, proof, anchored_hash=block["block_hash"], lookup=fake.lookup)["status"] == "pending"

    entry, notes = book.upgrade(1, fetch=fake.fetch, lookup=fake.lookup)
    assert entry["status"] == anchoring.STATUS_PENDING and any("still waiting" in n for n in notes)
    entry, notes = book.upgrade(1, fetch=fake.fetch, lookup=fake.lookup)
    assert any("less than" in n for n in notes)  # throttled: the calendars are not asked again so soon

    fake.mined = True
    entry, notes = book.upgrade(1, fetch=fake.fetch, lookup=fake.lookup, force=True)
    assert entry["status"] == anchoring.STATUS_CONFIRMED, notes
    assert entry["bitcoin"]["height"] == FakeBitcoin.HEIGHT and entry["bitcoin"]["checked_at"]
    header, proof = book.files(entry)
    result = anchoring.verify_files(header, proof, anchored_hash=block["block_hash"], chain_hash=block["block_hash"],
                                    lookup=fake.lookup)
    assert result["ok"] and result["status"] == anchoring.STATUS_CONFIRMED


def test_an_anchor_catches_history_rewritten_after_it(ledger, tmp_path, monkeypatch):
    fake = FakeBitcoin()
    book = anchoring.AnchorBook(tmp_path / "data")
    entry = book.create(ledger.read_blocks()[-1], calendars=("https://a.example",), submit=fake.submit)
    forged, _ = lc.simulate_attack(ledger.read_blocks(), "insider_rewrite", 1, ledger.private_pem)
    assert lc.verify_chain(forged, ledger.public_pem)["ok"]  # perfectly re-signed...
    header, proof = book.files(entry)
    result = anchoring.verify_files(header, proof, anchored_hash=entry["block_hash"],
                                    chain_hash=forged[1]["block_hash"], lookup=None)
    assert not result["ok"]  # ...but no longer the block that was anchored
    assert any(c["name"] == "The ledger still holds this exact block" and c["ok"] is False for c in result["checks"])


def test_no_calendar_answering_writes_nothing(ledger, tmp_path):
    def offline(url, digest):
        raise OSError("no network")
    book = anchoring.AnchorBook(tmp_path / "data")
    with pytest.raises(anchoring.AnchorError, match="online"):
        book.create(ledger.read_blocks()[-1], calendars=("https://a.example",), submit=offline)
    assert book.entries() == [] and not list(book.dir.glob("*.ots"))


def test_a_block_is_anchored_once_and_a_tampered_block_not_at_all(ledger, tmp_path):
    fake = FakeBitcoin()
    book = anchoring.AnchorBook(tmp_path / "data")
    block = ledger.read_blocks()[-1]
    book.create(block, calendars=("https://a.example",), submit=fake.submit)
    with pytest.raises(anchoring.AnchorError, match="already anchored"):
        book.create(block, calendars=("https://a.example",), submit=fake.submit)
    tampered = {**block, "merkle_root": "00" * 32}
    with pytest.raises(anchoring.AnchorError, match="tampered"):
        book.create(tampered, calendars=("https://a.example",), submit=fake.submit)


def test_upgrades_only_ask_whitelisted_calendars(ledger, tmp_path):
    asked = []

    def evil_submit(url, digest):
        ts = Timestamp(digest)
        ts.ops.add(OpSHA256()).attestations.add(PendingAttestation("https://evil.example.com"))
        return ts

    book = anchoring.AnchorBook(tmp_path / "data")
    book.create(ledger.read_blocks()[-1], calendars=("https://a.example",), submit=evil_submit)
    _, notes = book.upgrade(1, fetch=lambda uri, c: asked.append(uri), lookup=None)
    assert asked == [] and any("whitelist" in n for n in notes)


def test_the_index_file_survives_a_reload(ledger, tmp_path):
    fake = FakeBitcoin()
    book = anchoring.AnchorBook(tmp_path / "data")
    book.create(ledger.read_blocks()[-1], calendars=("https://a.example",), submit=fake.submit)
    again = anchoring.AnchorBook(tmp_path / "data")
    assert [e["block_index"] for e in again.entries()] == [1]
    assert json.loads(again.index_path.read_text(encoding="utf-8"))[0]["proof_file"] == "block-000001.header.json.ots"
