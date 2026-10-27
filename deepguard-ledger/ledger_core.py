"""
Deep-Guard Stage 3 — the provenance ledger (core data structures).

In plain words: a notebook that can only be written forwards. Every page
(a BLOCK) holds a few authenticity records, a single summary hash of
those records (the MERKLE ROOT), the hash of the page before it (the
PREVIOUS HASH) and Deep-Guard's signature. Change one letter on an old
page and that page's hashes stop adding up, and every page after it now
points at a hash that no longer exists — so the edit shows, exactly
where it was made.

This is the data-structure heart of a blockchain (hash-linked blocks,
Merkle trees, signatures) without mining, tokens or a cryptocurrency,
following the project blueprint's "blockchain vs. transparency log"
decision (section 3): the property the project needs is a tamper-evident,
append-only, independently checkable record, not Byzantine consensus
between strangers. The "no single party can silently rewrite it" half
comes from the witnesses (witness.py), which keep their own copies.

Nothing in this file knows about images, FastAPI or the detector. It only
stores, seals, proves and checks records, so it can be tested — and
checked offline by tools/verify_ledger.py — on its own.

Design rules, each one there for a reason:
  * Canonical JSON (sorted keys, no spaces, ASCII escapes) is the only
    thing ever hashed or signed, so Python here and JavaScript in the
    explorer page produce byte-identical input and the browser can redo
    every check itself.
  * No floats inside records: 0.1 + 0.2 prints differently in different
    languages. Numbers on the ledger are integers; percentages are strings.
  * Merkle leaves and inner nodes are hashed with different prefixes
    (0x00 / 0x01, as in Certificate Transparency, RFC 6962) so a pair of
    records can never be passed off as a single record.
  * An odd node at the end of a Merkle level is promoted unchanged, never
    duplicated (duplicating the last leaf lets two different record lists
    share one root — the Bitcoin CVE-2012-2459 problem).
  * The files are append-only: blocks are only ever appended to
    chain.jsonl, never rewritten. Fixing a mistake means appending a new
    record (a revocation or a dispute), not editing an old one.
"""

import copy
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

GENESIS_PREVIOUS_HASH = "0" * 64
LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

# The header is what the block hash (and therefore the signature) covers.
# The records are covered too — through merkle_root.
HEADER_FIELDS = ("index", "timestamp", "previous_hash", "merkle_root", "record_count", "signer_key_id")

WITNESS_STATEMENT = "deepguard-witness-v1|{index}|{block_hash}"

RECORD_TYPES = ("registration", "revocation", "dispute", "resolution")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _reject_floats(value, path: str = "record") -> None:
    if isinstance(value, float):
        raise TypeError(f"{path} is a float ({value!r}); ledger records only hold integers and strings")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _reject_floats(item, f"{path}[{i}]")


def canonical_bytes(value) -> bytes:
    """The one byte encoding that is ever hashed or signed."""
    _reject_floats(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_leaf_hash(record: dict) -> str:
    return sha256_hex(LEAF_PREFIX + canonical_bytes(record))


def _node_hash(left_hex: str, right_hex: str) -> str:
    return sha256_hex(NODE_PREFIX + bytes.fromhex(left_hex) + bytes.fromhex(right_hex))


def merkle_levels(records: list[dict]) -> list[list[str]]:
    """Every level of the Merkle tree, leaves first, root last. The explorer
    draws the tree from this."""
    if not records:
        return [[sha256_hex(b"")]]
    level = [record_leaf_hash(r) for r in records]
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(_node_hash(level[i], level[i + 1]) if i + 1 < len(level) else level[i])
        level = nxt
        levels.append(level)
    return levels


def merkle_root_hex(records: list[dict]) -> str:
    return merkle_levels(records)[-1][0]


def merkle_proof(records: list[dict], position: int) -> list[dict]:
    """The short list of sibling hashes that leads from one record up to the
    root. With it, anyone holding only the block header can check that the
    record is inside the block — without seeing the other records. Each
    step says which side the sibling sits on."""
    levels = merkle_levels(records)
    proof = []
    i = position
    for level in levels[:-1]:
        if i % 2 == 1:
            proof.append({"side": "left", "hash": level[i - 1]})
        elif i + 1 < len(level):
            proof.append({"side": "right", "hash": level[i + 1]})
        # else: this node was promoted unchanged, so this level adds no step
        i //= 2
    return proof


def verify_merkle_proof(leaf_hex: str, proof: list[dict], root_hex: str) -> bool:
    current = leaf_hex
    for step in proof:
        if step["side"] == "left":
            current = _node_hash(step["hash"], current)
        else:
            current = _node_hash(current, step["hash"])
    return current == root_hex


def block_header(block: dict) -> dict:
    return {field: block.get(field) for field in HEADER_FIELDS}


def block_hash_hex(block: dict) -> str:
    return sha256_hex(canonical_bytes(block_header(block)))


# ---------------------------------------------------------------------------
# Keys and signatures (Ed25519, the same algorithm fingerprint.py uses)
# ---------------------------------------------------------------------------

def generate_keypair() -> tuple[bytes, bytes]:
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem


def key_id(public_pem) -> str:
    """A short name for a public key: the first 16 hex digits of the SHA-256
    of its raw 32 bytes. Shown on every block so a change of signing key
    can never go unnoticed."""
    if isinstance(public_pem, str):
        public_pem = public_pem.encode("ascii")
    key = serialization.load_pem_public_key(public_pem)
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sha256_hex(raw)[:16]


def sign_hex(private_pem: bytes, data: bytes) -> str:
    return serialization.load_pem_private_key(private_pem, password=None).sign(data).hex()


def verify_hex(public_pem, data: bytes, signature_hex: str) -> bool:
    try:
        if isinstance(public_pem, str):
            public_pem = public_pem.encode("ascii")
        key = serialization.load_pem_public_key(public_pem)
        if not isinstance(key, Ed25519PublicKey):
            return False
        key.verify(bytes.fromhex(signature_hex), data)
        return True
    except Exception:  # noqa: BLE001 — any failure means "does not verify"
        return False


def sign_block(private_pem: bytes, block_hash: str) -> str:
    """Deep-Guard signs the 32 raw bytes of the block hash."""
    return sign_hex(private_pem, bytes.fromhex(block_hash))


def witness_statement(index: int, block_hash: str) -> bytes:
    return WITNESS_STATEMENT.format(index=index, block_hash=block_hash).encode("ascii")


def unsigned_record(record: dict) -> dict:
    """What a creator signs: the record minus its own signature."""
    return {k: v for k, v in record.items() if k != "creator_signature"}


def verify_creator_signature(record: dict) -> Optional[bool]:
    """True/False for records that carry a creator key; None for records
    that are signed only by Deep-Guard (resolutions)."""
    if "creator_public_key" not in record:
        return None
    return verify_hex(record["creator_public_key"], canonical_bytes(unsigned_record(record)),
                      record.get("creator_signature", ""))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Checking a chain (used by the server, the witnesses and the offline tool)
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check_block(block: dict, position: int, previous: Optional[dict], service_public_pem) -> dict:
    """Every check one block has to pass, each with a plain-words name, so a
    failure says exactly what broke instead of just "invalid"."""
    if block.get("_unreadable"):
        return {"index": position, "block_hash": None, "ok": False, "checks": [
            _check("Block can be read", False,
                   f"Line {block['_unreadable']} of chain.jsonl is not valid JSON any more (edited by hand?)")]}

    checks = []
    records = block.get("records") or []

    checks.append(_check("Position number is correct", block.get("index") == position,
                         f"says #{block.get('index')}, sits at position {position}"))

    expected_prev = GENESIS_PREVIOUS_HASH if previous is None else previous.get("block_hash")
    checks.append(_check("Points to the previous block", block.get("previous_hash") == expected_prev,
                         "" if block.get("previous_hash") == expected_prev else
                         f"stores previous_hash {str(block.get('previous_hash'))[:16]}…, "
                         f"but the block before it hashes to {str(expected_prev)[:16]}…"))

    try:
        root = merkle_root_hex(records)
        root_ok = root == block.get("merkle_root") and block.get("record_count") == len(records)
        detail = "" if root_ok else (
            f"records now hash to {root[:16]}…, block says {str(block.get('merkle_root'))[:16]}…"
            if root != block.get("merkle_root") else
            f"record_count says {block.get('record_count')}, block holds {len(records)}")
    except TypeError as exc:
        root_ok, detail = False, str(exc)
    checks.append(_check("Records still match the Merkle root", root_ok, detail))

    try:
        recomputed = block_hash_hex(block)
        hash_ok = recomputed == block.get("block_hash")
        detail = "" if hash_ok else f"contents hash to {recomputed[:16]}…, block says {str(block.get('block_hash'))[:16]}…"
    except TypeError as exc:
        hash_ok, detail = False, str(exc)
    checks.append(_check("Block hash matches its contents", hash_ok, detail))

    pinned = service_public_pem.decode("ascii") if isinstance(service_public_pem, bytes) else service_public_pem
    key_ok = block.get("signer_public_key") == pinned and block.get("signer_key_id") == key_id(pinned)
    sig_ok = key_ok and verify_hex(pinned, bytes.fromhex(str(block.get("block_hash") or "00")),
                                   block.get("signature", ""))
    checks.append(_check("Signed by Deep-Guard's key", sig_ok,
                         "" if sig_ok else ("signed by a different key" if not key_ok else
                                            "signature does not match this block hash")))

    bad_creators = []
    for i, record in enumerate(records):
        try:
            if verify_creator_signature(record) is False:
                bad_creators.append(i)
        except Exception:  # noqa: BLE001
            bad_creators.append(i)
    checks.append(_check("Every creator signature is valid", not bad_creators,
                         "" if not bad_creators else f"record(s) at position {bad_creators} fail"))

    if previous is not None and previous.get("timestamp") and block.get("timestamp"):
        checks.append(_check("Time only moves forward", block["timestamp"] >= previous["timestamp"],
                             "" if block["timestamp"] >= previous["timestamp"] else
                             "this block claims to be older than the one before it"))

    return {"index": position, "block_hash": block.get("block_hash"),
            "ok": all(c["ok"] for c in checks), "checks": checks}


def verify_chain(blocks: list[dict], service_public_pem) -> dict:
    results = []
    previous = None
    for position, block in enumerate(blocks):
        results.append(check_block(block, position, previous, service_public_pem))
        previous = block if not block.get("_unreadable") else {"block_hash": None}
    bad = [r["index"] for r in results if not r["ok"]]
    records = sum(len(b.get("records") or []) for b in blocks if not b.get("_unreadable"))
    return {
        "ok": not bad and bool(blocks),
        "block_count": len(blocks),
        "record_count": records,
        "first_bad_index": bad[0] if bad else None,
        "bad_indexes": bad,
        "blocks": results,
        "checked_at": now_iso(),
    }


def compare_with_witness(blocks: list[dict], witness_hashes: list[str]) -> dict:
    """A witness stored the hash of every block the first time it saw it.
    If Deep-Guard now shows a different hash at any of those positions,
    history was rewritten — even if the rewritten chain is internally
    perfect and properly signed."""
    first_mismatch = None
    for i, stored in enumerate(witness_hashes):
        current = blocks[i].get("block_hash") if i < len(blocks) else None
        if current != stored:
            first_mismatch = i
            break
    return {
        "agrees": first_mismatch is None and len(witness_hashes) <= len(blocks),
        "witness_height": len(witness_hashes),
        "first_mismatch": first_mismatch,
        "missing_blocks": max(len(blocks) - len(witness_hashes), 0),
    }


# ---------------------------------------------------------------------------
# The ledger itself (files on disk)
# ---------------------------------------------------------------------------

class Ledger:
    """chain.jsonl — one sealed block per line, only ever appended to.
    pending.jsonl — records waiting to be sealed into the next block.
    cosignatures.json — witness signatures collected after sealing
                         (they arrive after a block exists, so they cannot
                         live inside it)."""

    def __init__(self, data_dir: Path, service_private_pem: bytes, service_public_pem: bytes):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chain_path = self.dir / "chain.jsonl"
        self.pending_path = self.dir / "pending.jsonl"
        self.cosign_path = self.dir / "cosignatures.json"
        self.private_pem = service_private_pem
        self.public_pem = service_public_pem
        self.public_pem_str = service_public_pem.decode("ascii")
        self.key_id = key_id(service_public_pem)
        self.lock = threading.RLock()
        with self.lock:
            if not self.chain_path.exists() or not self.chain_path.read_text(encoding="utf-8").strip():
                self._append_block([])  # genesis: block #0, no records

    # ----------------------------------------------------------------- read
    def read_blocks(self) -> list[dict]:
        """Read straight from disk every time, so a hand edit of chain.jsonl
        (the live tamper demo) is what gets checked, not a cached copy."""
        if not self.chain_path.exists():
            return []
        blocks = []
        for line_no, line in enumerate(self.chain_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                blocks.append({"_unreadable": line_no, "raw": line[:300]})
        return blocks

    def read_pending(self) -> list[dict]:
        if not self.pending_path.exists():
            return []
        return [json.loads(line) for line in self.pending_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def read_cosignatures(self) -> dict:
        if not self.cosign_path.exists():
            return {}
        try:
            return json.loads(self.cosign_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---------------------------------------------------------------- write
    def add_pending(self, record: dict) -> None:
        if record.get("type") not in RECORD_TYPES:
            raise ValueError(f"unknown record type {record.get('type')!r}")
        canonical_bytes(record)  # refuses floats before anything is written
        with self.lock, self.pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_block(self, records: list[dict]) -> dict:
        blocks = self.read_blocks()
        previous = blocks[-1] if blocks else None
        header = {
            "index": len(blocks),
            "timestamp": now_iso(),
            "previous_hash": previous.get("block_hash") if previous else GENESIS_PREVIOUS_HASH,
            "merkle_root": merkle_root_hex(records),
            "record_count": len(records),
            "signer_key_id": self.key_id,
        }
        block_hash = sha256_hex(canonical_bytes(header))
        block = {
            **header,
            "block_hash": block_hash,
            "signature": sign_block(self.private_pem, block_hash),
            "signer_public_key": self.public_pem_str,
            "records": records,
        }
        with self.chain_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(block, sort_keys=True) + "\n")
        return block

    def seal(self) -> Optional[dict]:
        """Turns every pending record into one new signed block."""
        with self.lock:
            pending = self.read_pending()
            if not pending:
                return None
            block = self._append_block(pending)
            self.pending_path.write_text("", encoding="utf-8")
            return block

    def save_cosignatures(self, index: int, entries: list[dict]) -> None:
        with self.lock:
            stored = self.read_cosignatures()
            current = {e["witness_key_id"]: e for e in stored.get(str(index), [])}
            for entry in entries:
                current[entry["witness_key_id"]] = entry
            stored[str(index)] = list(current.values())
            self.cosign_path.write_text(json.dumps(stored, indent=1, sort_keys=True), encoding="utf-8")

    # ---------------------------------------------------------------- query
    def verify(self, blocks: Optional[list[dict]] = None) -> dict:
        return verify_chain(self.read_blocks() if blocks is None else blocks, self.public_pem)

    def iter_records(self, blocks: Optional[list[dict]] = None):
        """(record, block_index or None if still pending, position in block)."""
        for block in (self.read_blocks() if blocks is None else blocks):
            if block.get("_unreadable"):
                continue
            for position, record in enumerate(block.get("records") or []):
                yield record, block.get("index"), position
        for position, record in enumerate(self.read_pending()):
            yield record, None, position

    def find_record(self, record_id: str, blocks: Optional[list[dict]] = None):
        for record, block_index, position in self.iter_records(blocks):
            if record.get("record_id") == record_id:
                return record, block_index, position
        return None, None, None

    def status_of(self, record_id: str, blocks: Optional[list[dict]] = None) -> dict:
        """Later records that talk about this one: a revocation by its
        creator, disputes filed by anyone, and Deep-Guard's resolutions."""
        status = {"revoked": None, "disputes": [], "resolutions": []}
        for record, block_index, _ in self.iter_records(blocks):
            if record.get("target_record_id") != record_id:
                continue
            entry = {**record, "block_index": block_index}
            if record["type"] == "revocation":
                status["revoked"] = entry
            elif record["type"] == "dispute":
                status["disputes"].append(entry)
            elif record["type"] == "resolution":
                status["resolutions"].append(entry)
        return status

    def proof_bundle(self, record_id: str, blocks: Optional[list[dict]] = None) -> Optional[dict]:
        """Everything needed to check one record without trusting the
        server: the record, its Merkle path, the block header, Deep-Guard's
        signature and public key, and any witness co-signatures."""
        blocks = self.read_blocks() if blocks is None else blocks
        record, block_index, position = self.find_record(record_id, blocks)
        if record is None:
            return None
        if block_index is None:
            return {"record": record, "sealed": False}
        block = blocks[block_index]
        return {
            "sealed": True,
            "record": record,
            "leaf_hash": record_leaf_hash(record),
            "position": position,
            "merkle_proof": merkle_proof(block["records"], position),
            "block_header": block_header(block),
            "block_hash": block["block_hash"],
            "signature": block["signature"],
            "signer_public_key": block["signer_public_key"],
            "witness_cosignatures": self.read_cosignatures().get(str(block_index), []),
        }


def verify_receipt(receipt: dict, service_public_pem) -> dict:
    """Checks a downloaded receipt on its own — no ledger, no server.
    Used by tools/verify_ledger.py; the explorer page does the same in JS."""
    record = receipt["record"]
    header = receipt["block_header"]
    pinned = service_public_pem.decode("ascii") if isinstance(service_public_pem, bytes) else service_public_pem
    leaf = record_leaf_hash(record)
    checks = [
        _check("Record hashes to the leaf in the receipt", leaf == receipt.get("leaf_hash")),
        _check("Merkle path leads to the block's root", verify_merkle_proof(leaf, receipt["merkle_proof"], header["merkle_root"])),
        _check("Block hash matches the header", sha256_hex(canonical_bytes(header)) == receipt["block_hash"]),
        _check("Header names the expected signing key", header.get("signer_key_id") == key_id(pinned)),
        _check("Deep-Guard's signature is valid", verify_hex(pinned, bytes.fromhex(receipt["block_hash"]), receipt["signature"])),
    ]
    creator = verify_creator_signature(record)
    if creator is not None:
        checks.append(_check("Creator's signature is valid", creator))
    for cosig in receipt.get("witness_cosignatures", []):
        ok = verify_hex(cosig["witness_public_key"], witness_statement(header["index"], receipt["block_hash"]),
                        cosig["signature"])
        checks.append(_check(f"{cosig['witness']} co-signed this block", ok))
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# ---------------------------------------------------------------------------
# Tamper lab: attacks run on a COPY of the chain, never the real files
# ---------------------------------------------------------------------------

ATTACKS = {
    "edit_record": "Change one word inside an old record and leave everything else alone.",
    "edit_and_rehash": "Change a record, then recompute that block's Merkle root and block hash "
                       "so the block looks consistent — without Deep-Guard's private key.",
    "delete_block": "Remove a whole block from the middle of the chain.",
    "insider_rewrite": "Deep-Guard itself (holding the real key) quietly erases a record from an old block "
                       "and re-signs that block and every block after it, so the chain checks out perfectly "
                       "on its own. (Editing a record instead would break its creator's signature.)",
}


def simulate_attack(blocks: list[dict], attack: str, index: int, private_pem: bytes) -> tuple[list[dict], str]:
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}")
    if not 0 <= index < len(blocks):
        raise ValueError(f"there is no block #{index}")
    if attack == "delete_block" and index == 0:
        raise ValueError("pick a block after #0 to delete")
    chain = copy.deepcopy(blocks)
    target = chain[index]

    def edit(block) -> str:
        records = block.get("records") or []
        if records:
            record = records[0]
            field = "title" if "title" in record else "type"
            old = record.get(field)
            record[field] = f"{old} (forged)"
            return f"changed record {record.get('record_id', '?')[:8]}… field '{field}' from '{old}' to '{record[field]}'"
        old = block["timestamp"]
        block["timestamp"] = "2001-01-01T00:00:00.000+00:00"
        return f"changed the block's timestamp from {old} to 2001-01-01"

    if attack == "edit_record":
        what = edit(target)
        return chain, f"In block #{index}: {what}."

    if attack == "edit_and_rehash":
        what = edit(target)
        target["merkle_root"] = merkle_root_hex(target["records"])
        target["block_hash"] = block_hash_hex(target)
        return chain, (f"In block #{index}: {what}, then recomputed its Merkle root and block hash. "
                       f"The attacker has no private key, so the old signature stays.")

    if attack == "delete_block":
        del chain[index]
        return chain, f"Removed block #{index} entirely; the blocks after it moved up one place."

    # insider_rewrite: a perfectly re-signed alternative history. Editing a
    # record would break its creator's signature, which Deep-Guard cannot
    # forge — but erasing a whole record (or backdating an empty block)
    # leaves nothing inside the chain to complain.
    records = target.get("records") or []
    if records:
        gone = records.pop(0)
        target["record_count"] = len(records)
        what = f"erased record {gone.get('record_id', '?')[:8]}… ('{gone.get('title') or gone.get('type')}')"
    else:
        target["timestamp"] = min(target["timestamp"], "2001-01-01T00:00:00.000+00:00") if index == 0 else \
            chain[index - 1]["timestamp"]
        what = "backdated the block"
    for i in range(index, len(chain)):
        block = chain[i]
        if i > index:
            block["previous_hash"] = chain[i - 1]["block_hash"]
        block["merkle_root"] = merkle_root_hex(block["records"])
        block["block_hash"] = block_hash_hex(block)
        block["signature"] = sign_block(private_pem, block["block_hash"])
    return chain, (f"In block #{index}: {what}, then re-hashed and re-signed block #{index} and every "
                   f"block after it with Deep-Guard's real key.")
