"""
Deep-Guard Stage 4 — anchoring the ledger to Bitcoin with OpenTimestamps.

In plain words: the witnesses stop Deep-Guard from quietly rewriting history
on its own, but Deep-Guard and its witnesses together could still agree to
backdate it. An anchor closes that gap. The newest block's hash is sent to the
public OpenTimestamps calendars, which fold it together with thousands of
other people's hashes into one hash and write that into a Bitcoin
transaction. Once the transaction is mined, the block provably existed before
that Bitcoin block, and nobody can change Bitcoin's history to say otherwise.

Why anchoring the newest block is enough: every block stores the hash of the
block before it, so the newest block's hash commits to the whole chain.

What leaves this machine: only the 32-byte block hash, salted with a random
16-byte nonce first (as the official OpenTimestamps client does), so the
calendars never learn which block it was. No record, title, picture or key.

The "file" being timestamped is the block's canonical header: block_hash is
SHA-256(canonical header bytes), so saving those bytes as
block-NNNNNN.header.json next to its .ots proof gives anyone two files they
can check with standard OpenTimestamps tools, including opentimestamps.org,
without any Deep-Guard code.

A proof starts "pending" (a calendar's promise) and becomes "confirmed" a few
hours later, once the calendar's Bitcoin transaction is mined and the proof
is upgraded with the path into that block. Confirming does not trust the
calendar: the proof's operations are replayed and the result must equal the
Merkle root of that Bitcoin block, looked up on independent block explorers
that have to agree with each other.

Nothing here knows about FastAPI or images; ledger_app.py wires it in and
tools/verify_ledger.py checks anchors offline.
"""

import hashlib
import io
import json
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from opentimestamps.calendar import DEFAULT_CALENDAR_WHITELIST, CommitmentNotFoundError, RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import BinaryOp, OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

import ledger_core

# The aggregators the official client submits to by default. Each one forwards
# to its own calendar (alice/bob.btc.calendar.opentimestamps.org,
# finney.calendar.eternitywall.com, btc.calendar.catallaxy.com), so one
# answer is enough; more answers mean the proof survives a calendar vanishing.
CALENDARS = tuple(u.strip() for u in os.environ.get(
    "DEEPGUARD_OTS_CALENDARS",
    "https://a.pool.opentimestamps.org,https://b.pool.opentimestamps.org,"
    "https://a.pool.eternitywall.com,https://ots.btc.catallaxy.com").split(",") if u.strip())

USER_AGENT = "deepguard-ledger/1.0"
NETWORK_TIMEOUT = 15  # seconds per request; the calendars answered in 2-4 s when measured
MIN_UPGRADE_INTERVAL = 60  # seconds between asking the calendars about the same anchor

# A proof names the calendar to ask for its upgrade. Only calendars on the
# official client's whitelist are ever contacted, so a proof cannot make this
# server fetch an arbitrary address.
UPGRADE_WHITELIST = DEFAULT_CALENDAR_WHITELIST

STATUS_PENDING = "pending"        # in the calendars, not yet in a mined Bitcoin block
STATUS_CLAIMED = "claimed"        # names a Bitcoin block, not looked up yet (offline, or explorers unreachable)
STATUS_CONFIRMED = "confirmed"    # the proof reaches a Bitcoin block's Merkle root, checked on explorers
STATUS_FAILED = "failed"          # the proof claims a Bitcoin block it does not reach


class AnchorError(Exception):
    """Something a person should be told in plain words (offline, nothing to anchor, ...)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def header_file_name(index: int) -> str:
    return f"block-{index:06d}.header.json"


def header_bytes(block: dict) -> bytes:
    """The exact bytes whose SHA-256 is the block hash."""
    return ledger_core.canonical_bytes(ledger_core.block_header(block))


# ---------------------------------------------------------------------------
# Proof files
# ---------------------------------------------------------------------------

def serialize_proof(dtf: DetachedTimestampFile) -> bytes:
    buffer = io.BytesIO()
    dtf.serialize(StreamSerializationContext(buffer))
    return buffer.getvalue()


def load_proof(data: bytes) -> DetachedTimestampFile:
    return DetachedTimestampFile.deserialize(StreamDeserializationContext(io.BytesIO(data)))


def _nodes(timestamp: Timestamp):
    yield timestamp
    for child in timestamp.ops.values():
        yield from _nodes(child)


def attestation_paths(timestamp: Timestamp) -> list[dict]:
    """Every route from the block hash to an attestation, as plain steps the
    explorer page can replay itself: [{"op": "append", "arg": hex}, {"op":
    "sha256"}, ...]. Replaying them on the block hash must give `msg`, which
    for a Bitcoin attestation must equal that block's Merkle root."""
    paths = []

    def walk(node: Timestamp, steps: list):
        for attestation in sorted(node.attestations, key=lambda a: a.__class__.__name__):
            entry = {"msg": node.msg.hex(), "ops": list(steps)}
            if isinstance(attestation, BitcoinBlockHeaderAttestation):
                entry["attestation"] = {"type": "bitcoin", "height": attestation.height}
            elif isinstance(attestation, PendingAttestation):
                entry["attestation"] = {"type": "pending", "uri": attestation.uri}
            else:
                entry["attestation"] = {"type": attestation.__class__.__name__}
            paths.append(entry)
        for op, child in node.ops.items():
            step = {"op": op.TAG_NAME}
            if isinstance(op, BinaryOp):  # append/prepend carry the bytes they add
                step["arg"] = op[0].hex()
            walk(child, steps + [step])

    walk(timestamp, [])
    return paths


def bitcoin_attestations(timestamp: Timestamp) -> list[tuple[bytes, int]]:
    return [(msg, att.height) for msg, att in timestamp.all_attestations()
            if isinstance(att, BitcoinBlockHeaderAttestation)]


def pending_calendars(timestamp: Timestamp) -> list[str]:
    return sorted({att.uri for _, att in timestamp.all_attestations() if isinstance(att, PendingAttestation)})


# ---------------------------------------------------------------------------
# Talking to the calendars
# ---------------------------------------------------------------------------

def submit_to_calendar(url: str, digest: bytes) -> Timestamp:
    return RemoteCalendar(url, user_agent=USER_AGENT).submit(digest, timeout=NETWORK_TIMEOUT)


def fetch_upgrade(url: str, commitment: bytes) -> Optional[Timestamp]:
    """The rest of the proof from a pending commitment onward, or None while
    the calendar is still waiting for its Bitcoin transaction to be mined."""
    try:
        return RemoteCalendar(url, user_agent=USER_AGENT).get_timestamp(commitment, timeout=NETWORK_TIMEOUT)
    except CommitmentNotFoundError:
        return None


def stamp(digest: bytes, calendars=CALENDARS, submit: Callable = submit_to_calendar) -> tuple[DetachedTimestampFile, list]:
    """Builds the proof for one 32-byte digest: digest -> append(random
    nonce) -> sha256 -> each calendar's own operations -> its promise."""
    root = Timestamp(digest)
    commitment = root.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
    answers = []
    for url in calendars:
        try:
            commitment.merge(submit(url, commitment.msg))
            answers.append({"url": url, "ok": True})
        except Exception as exc:  # noqa: BLE001 — one calendar failing is normal; report it, keep going
            answers.append({"url": url, "ok": False, "error": f"{exc.__class__.__name__}: {str(exc)[:160]}"})
    if not any(a["ok"] for a in answers):
        raise AnchorError("No OpenTimestamps calendar answered. Is this computer online? "
                          + "; ".join(f"{a['url']}: {a['error']}" for a in answers))
    return DetachedTimestampFile(OpSHA256(), root), answers


def upgrade_proof(dtf: DetachedTimestampFile, fetch: Callable = fetch_upgrade) -> tuple[bool, list]:
    """Asks each calendar the proof is waiting on for the rest of its path.
    Returns (changed, notes). Calendars not on the whitelist are skipped."""
    changed, notes = False, []
    for node in list(_nodes(dtf.timestamp)):
        for attestation in list(node.attestations):
            if not isinstance(attestation, PendingAttestation):
                continue
            if attestation.uri not in UPGRADE_WHITELIST:
                notes.append(f"Skipped {attestation.uri}: not on the OpenTimestamps calendar whitelist.")
                continue
            try:
                upgraded = fetch(attestation.uri, node.msg)
            except Exception as exc:  # noqa: BLE001 — offline or a calendar hiccup: try again later
                notes.append(f"{attestation.uri}: {exc.__class__.__name__}")
                continue
            if upgraded is None:
                notes.append(f"{attestation.uri}: still waiting for its Bitcoin transaction to be mined.")
                continue
            node.merge(upgraded)
            changed = True
            notes.append(f"{attestation.uri}: upgraded.")
    return changed, notes


# ---------------------------------------------------------------------------
# Looking up Bitcoin blocks
# ---------------------------------------------------------------------------

def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT) as response:
        return response.read(200_000).decode("utf-8")


def _esplora(base: str):
    def lookup(height: int) -> dict:
        block_hash = _get(f"{base}/block-height/{height}").strip()
        block = json.loads(_get(f"{base}/block/{block_hash}"))
        return {"block_hash": block["id"], "merkle_root": block["merkle_root"], "time": int(block["timestamp"])}
    return lookup


def _blockcypher(height: int) -> dict:
    block = json.loads(_get(f"https://api.blockcypher.com/v1/btc/main/blocks/{height}?txstart=0&limit=1"))
    when = datetime.fromisoformat(block["time"].replace("Z", "+00:00"))
    return {"block_hash": block["hash"], "merkle_root": block["mrkl_root"], "time": int(when.timestamp())}


# Explorers run by different organisations. mempool.space did not answer
# from the network this was built on, which is exactly why there are three.
EXPLORERS = (
    ("blockstream.info", _esplora("https://blockstream.info/api")),
    ("blockcypher.com", _blockcypher),
    ("mempool.space", _esplora("https://mempool.space/api")),
)
REQUIRED_AGREEING_EXPLORERS = 2


def bitcoin_block(height: int, explorers=EXPLORERS) -> dict:
    """The block at `height` as two independent explorers report it. Raises
    AnchorError if they disagree, or if none can be reached. One answer alone
    is returned but marked single_source, so the page can say so."""
    answers, problems = [], []
    for name, lookup in explorers:
        try:
            answers.append((name, lookup(height)))
        except Exception as exc:  # noqa: BLE001 — unreachable explorer: try the next
            problems.append(f"{name}: {exc.__class__.__name__}")
            continue
        if len(answers) >= REQUIRED_AGREEING_EXPLORERS:
            break
    if not answers:
        raise AnchorError(f"No block explorer could be reached ({'; '.join(problems)}). Is this computer online?")
    first = answers[0][1]
    for name, other in answers[1:]:
        if (other["block_hash"], other["merkle_root"]) != (first["block_hash"], first["merkle_root"]):
            raise AnchorError(f"Block explorers disagree about Bitcoin block {height} ({answers[0][0]} vs {name}). "
                              "Not trusting either.")
    return {"height": height, **first, "time_utc": datetime.fromtimestamp(first["time"], timezone.utc).isoformat(),
            "sources": [name for name, _ in answers], "single_source": len(answers) < REQUIRED_AGREEING_EXPLORERS}


def merkle_root_matches(msg: bytes, merkle_root_hex: str) -> bool:
    """Explorers print a Merkle root byte-reversed; the proof ends on the root
    as it sits inside the block header. Checked on OpenTimestamps' own example
    proof (block 358391) against two explorers before this line was written."""
    return msg == bytes.fromhex(merkle_root_hex)[::-1]


# ---------------------------------------------------------------------------
# Checking one anchor
# ---------------------------------------------------------------------------

def _check(name: str, ok: Optional[bool], detail: str = "") -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def verify_files(header: bytes, proof: bytes, anchored_hash: Optional[str] = None,
                 chain_hash: Optional[str] = None, lookup: Optional[Callable] = bitcoin_block) -> dict:
    """Every check one anchor has to pass, each with a plain-words name.
    lookup(height) -> bitcoin_block-style dict; None skips the Bitcoin step
    (an offline check of everything up to it)."""
    checks = []
    digest = hashlib.sha256(header).digest()
    if anchored_hash is not None:
        checks.append(_check("Header file hashes to the anchored block hash", digest.hex() == anchored_hash,
                             "" if digest.hex() == anchored_hash else
                             f"the header file hashes to {digest.hex()[:16]}…, the anchor names {anchored_hash[:16]}…"))
    if chain_hash is not None:
        same = chain_hash == digest.hex()
        checks.append(_check("The ledger still holds this exact block", same, "" if same else
                             f"the chain's block now hashes to {str(chain_hash)[:16]}…: history was changed after "
                             f"it was anchored"))
    try:
        dtf = load_proof(proof)
    except Exception as exc:  # noqa: BLE001 — a broken proof file is a finding, not a crash
        checks.append(_check("Proof file can be read", False, f"{exc.__class__.__name__}: {exc}"))
        return {"ok": False, "status": STATUS_FAILED, "checks": checks, "bitcoin": None, "paths": []}
    checks.append(_check("Proof is for this header", dtf.file_digest == digest,
                         "" if dtf.file_digest == digest else "the proof was made for different bytes"))

    bitcoin = None
    attestations = bitcoin_attestations(dtf.timestamp)
    status = STATUS_PENDING
    if attestations and lookup is None:
        heights = sorted({h for _, h in attestations})
        status = STATUS_CLAIMED
        checks.append(_check("Proof reaches a Bitcoin block", None,
                             f"claims Bitcoin block(s) {heights}; not looked up (offline check)"))
    elif attestations:
        status = STATUS_CLAIMED
        for msg, height in sorted(attestations, key=lambda a: a[1]):
            try:
                block = lookup(height)
            except AnchorError as exc:
                checks.append(_check(f"Proof reaches Bitcoin block #{height}", None, str(exc)))
                continue
            ok = merkle_root_matches(msg, block["merkle_root"])
            sources = " and ".join(block["sources"])
            checks.append(_check(f"Proof reaches Bitcoin block #{height}'s Merkle root", ok,
                                 (f"checked on {sources}" + (" (only one explorer answered)" if block.get("single_source") else ""))
                                 if ok else f"the proof ends on {msg.hex()[:16]}…, {sources} say the block's root is "
                                            f"{block['merkle_root'][:16]}… (reversed)"))
            if ok and (bitcoin is None or height < bitcoin["height"]):
                bitcoin = block
            if not ok:
                status = STATUS_FAILED
        if bitcoin is not None and status != STATUS_FAILED:
            status = STATUS_CONFIRMED
    else:
        calendars = pending_calendars(dtf.timestamp)
        checks.append(_check("Waiting for Bitcoin", None,
                             f"{len(calendars)} calendar(s) hold the commitment; a Bitcoin transaction "
                             f"usually confirms within a few hours"))

    ok = all(c["ok"] is not False for c in checks)
    if not ok:
        status = STATUS_FAILED
    return {"ok": ok, "status": status, "checks": checks, "bitcoin": bitcoin,
            "paths": attestation_paths(dtf.timestamp)}


# ---------------------------------------------------------------------------
# The anchors kept beside the ledger
# ---------------------------------------------------------------------------

class AnchorBook:
    """ledger_data/anchors/ — anchors.json (one entry per anchored block) plus,
    for each, the header bytes and the .ots proof. Kept beside the chain like
    the witness co-signatures: they are proofs ABOUT blocks, so they cannot
    live inside them, and a proof keeps growing (pending -> confirmed) after
    its block is long sealed."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "anchors"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "anchors.json"
        self.lock = threading.RLock()

    # ----------------------------------------------------------------- read
    def entries(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def entry(self, index: int) -> Optional[dict]:
        return next((e for e in self.entries() if e["block_index"] == index), None)

    def header_path(self, entry: dict) -> Path:
        return self.dir / entry["header_file"]

    def proof_path(self, entry: dict) -> Path:
        return self.dir / entry["proof_file"]

    def files(self, entry: dict) -> tuple[bytes, bytes]:
        return self.header_path(entry).read_bytes(), self.proof_path(entry).read_bytes()

    # ---------------------------------------------------------------- write
    def _save(self, entries: list[dict]) -> None:
        entries = sorted(entries, key=lambda e: e["block_index"])
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.index_path)

    def _replace(self, updated: dict) -> None:
        entries = [e for e in self.entries() if e["block_index"] != updated["block_index"]]
        self._save(entries + [updated])

    def create(self, block: dict, calendars=CALENDARS, submit: Callable = submit_to_calendar) -> dict:
        index = block.get("index")
        if not isinstance(index, int):
            raise AnchorError("That block can't be read, so it can't be anchored.")
        header = header_bytes(block)
        if hashlib.sha256(header).hexdigest() != block.get("block_hash"):
            raise AnchorError(f"Block #{index} fails its own hash check; anchoring it would anchor a tampered block.")
        with self.lock:
            if self.entry(index) is not None:
                raise AnchorError(f"Block #{index} is already anchored.")
            dtf, answers = stamp(bytes.fromhex(block["block_hash"]), calendars, submit)
            entry = {
                "block_index": index,
                "block_hash": block["block_hash"],
                "anchored_at": _now_iso(),
                "calendars": answers,
                "status": STATUS_PENDING,
                "bitcoin": None,
                "last_upgrade_attempt": None,
                "header_file": header_file_name(index),
                "proof_file": header_file_name(index) + ".ots",
            }
            self.header_path(entry).write_bytes(header)
            self.proof_path(entry).write_bytes(serialize_proof(dtf))
            self._replace(entry)
            return entry

    def upgrade(self, index: int, fetch: Callable = fetch_upgrade, lookup: Optional[Callable] = bitcoin_block,
                force: bool = False) -> tuple[dict, list]:
        """Asks the calendars for the rest of a pending proof, then checks any
        Bitcoin attestation it now has against the explorers."""
        with self.lock:
            entry = self.entry(index)
            if entry is None:
                raise AnchorError(f"Block #{index} has no anchor.")
            notes = []
            if entry["status"] == STATUS_PENDING:
                last = entry.get("last_upgrade_attempt")
                recent = last and (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() < MIN_UPGRADE_INTERVAL
                if recent and not force:
                    return entry, [f"Asked the calendars less than {MIN_UPGRADE_INTERVAL} s ago; try again in a minute."]
                header, proof = self.files(entry)
                dtf = load_proof(proof)
                changed, notes = upgrade_proof(dtf, fetch)
                entry["last_upgrade_attempt"] = _now_iso()
                if changed:
                    self.proof_path(entry).write_bytes(serialize_proof(dtf))
            header, proof = self.files(entry)
            if bitcoin_attestations(load_proof(proof).timestamp) and lookup is not None:
                result = verify_files(header, proof, anchored_hash=entry["block_hash"], lookup=lookup)
                if result["status"] in (STATUS_CLAIMED, STATUS_CONFIRMED, STATUS_FAILED):
                    entry["status"] = result["status"]
                    entry["bitcoin"] = {**result["bitcoin"], "checked_at": _now_iso()} if result["bitcoin"] else None
                notes.append("Checked against Bitcoin: " + result["status"] + ".")
            self._replace(entry)
            return entry, notes
