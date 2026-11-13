"""
A witness node — an independent copy-keeper of the ledger.

In plain words: Deep-Guard holds the key that signs the ledger, so on its
own nothing stops Deep-Guard from quietly writing a new, perfectly signed
history. A witness is a second notary who receives every new block, checks
it, writes down its hash in its OWN notebook and signs a short statement
"I saw block #N with hash H". If Deep-Guard later shows a different block
#N, the witness's notebook disagrees and the rewrite is caught. This is how
Certificate Transparency and Sigstore keep their logs honest, and it is the
"multi-node" stretch goal from the blueprint (section 3) in its simplest
form: each witness is a separate process with its own key and its own files.

A witness never accepts a block that does not follow from the last one it
stored (wrong previous hash, wrong Merkle root, bad signature, a different
hash for a block it already has). It answers with HTTP 409 and keeps the
complaint in its alarm log, which the explorer page shows.

Run on its own (run_deepguard.py starts two of these for you):
    set DEEPGUARD_WITNESS_NAME=Witness A
    python -m uvicorn witness:app --port 8101
"""

import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import ledger_core
import public_mode

HERE = Path(__file__).resolve().parent
NAME = os.environ.get("DEEPGUARD_WITNESS_NAME", "Witness A")
DATA_DIR = Path(os.environ.get("DEEPGUARD_WITNESS_DIR",
                               HERE / "ledger_data" / "witnesses" / ledger_core.sha256_hex(NAME.encode())[:8]))
# The witness checks every block against Deep-Guard's public key, read from
# the bouncer's key file — configured, never taken from the blocks it is
# checking.
SERVICE_PUBLIC_KEY_PATH = Path(os.environ.get(
    "DEEPGUARD_SERVICE_PUBLIC_KEY", HERE.parent / "deepguard-bouncer" / "models" / "demo_signer_public.pem"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
KEY_PATH = DATA_DIR / "witness_private.pem"
LOG_PATH = DATA_DIR / "seen_blocks.jsonl"
ALARM_PATH = DATA_DIR / "alarms.jsonl"
LOCK = threading.Lock()

if KEY_PATH.exists():
    PRIVATE_PEM = KEY_PATH.read_bytes()
    from cryptography.hazmat.primitives import serialization
    PUBLIC_PEM = serialization.load_pem_private_key(PRIVATE_PEM, password=None).public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
else:
    PRIVATE_PEM, PUBLIC_PEM = ledger_core.generate_keypair()
    KEY_PATH.write_bytes(PRIVATE_PEM)

app = FastAPI(title=f"Deep-Guard {NAME}")
# Open to every origin on purpose: the explorer page asks the witness
# directly from the browser, so the comparison does not go through (and
# does not have to trust) the Deep-Guard server.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _seen() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(l) for l in LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _alarms() -> list[dict]:
    if not ALARM_PATH.exists():
        return []
    return [json.loads(l) for l in ALARM_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _raise_alarm(message: str, index) -> None:
    entry = {"at": ledger_core.now_iso(), "block_index": index, "message": message}
    with ALARM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    raise HTTPException(status_code=409, detail=entry)


def _identity() -> dict:
    return {"witness": NAME, "witness_public_key": PUBLIC_PEM.decode("ascii"),
            "witness_key_id": ledger_core.key_id(PUBLIC_PEM)}


@app.get("/witness/info")
def info():
    seen = _seen()
    return {**_identity(), "height": len(seen), "last_hash": seen[-1]["block_hash"] if seen else None,
            "alarms": _alarms()[-10:]}


@app.get("/witness/hashes")
def hashes():
    seen = _seen()
    return {**_identity(), "hashes": [s["block_hash"] for s in seen],
            "cosignatures": [s["signature"] for s in seen], "alarms": _alarms()[-10:]}


@app.post("/witness/blocks")
async def receive_blocks(request: Request):
    if public_mode.is_public_request(request.headers):
        # `run_deepguard.py --public` gives each witness a tunnel of its own
        # so visitors' browsers can ask it directly. That tunnel is for
        # reading: blocks come only from the ledger on this laptop, or a
        # visitor could post a forged block and fill the alarm log that the
        # explorer shows with false "history was rewritten" alarms.
        raise HTTPException(status_code=403, detail="Blocks reach this witness only from the Deep-Guard ledger on "
                                                    "its own laptop, not over the public link.")
    body = await request.json()
    blocks = body.get("blocks") or []
    if not SERVICE_PUBLIC_KEY_PATH.exists():
        raise HTTPException(status_code=503, detail="Deep-Guard's public key file does not exist yet.")
    service_pem = SERVICE_PUBLIC_KEY_PATH.read_bytes()
    accepted = []
    with LOCK:
        seen = _seen()
        for block in blocks:
            index = block.get("index")
            if not isinstance(index, int):
                _raise_alarm("A block arrived without a position number.", None)
            if index < len(seen):
                if seen[index]["block_hash"] != block.get("block_hash"):
                    _raise_alarm(f"Deep-Guard showed a DIFFERENT block #{index} than the one this witness "
                                 f"already signed (had {seen[index]['block_hash'][:16]}…, now "
                                 f"{str(block.get('block_hash'))[:16]}…). History was rewritten.", index)
                accepted.append({**_identity(), "index": index, "block_hash": seen[index]["block_hash"],
                                 "signature": seen[index]["signature"]})
                continue
            if index > len(seen):
                raise HTTPException(status_code=409, detail={
                    "message": f"Missing blocks: this witness has {len(seen)}, send #{len(seen)} first.",
                    "block_index": index})
            previous = {"block_hash": seen[-1]["block_hash"], "timestamp": seen[-1]["timestamp"]} if seen else None
            result = ledger_core.check_block(block, index, previous, service_pem)
            if not result["ok"]:
                failed = "; ".join(f"{c['name']}: {c['detail'] or 'failed'}" for c in result["checks"] if not c["ok"])
                _raise_alarm(f"Block #{index} was refused: {failed}", index)
            signature = ledger_core.sign_hex(PRIVATE_PEM, ledger_core.witness_statement(index, block["block_hash"]))
            entry = {"index": index, "block_hash": block["block_hash"], "timestamp": block["timestamp"],
                     "signature": signature, "seen_at": ledger_core.now_iso()}
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            seen.append(entry)
            accepted.append({**_identity(), "index": index, "block_hash": block["block_hash"], "signature": signature})
    return {**_identity(), "height": len(seen), "cosignatures": accepted}
