"""
Deep-Guard + the provenance ledger (Stage 3).

This file does not copy or change the bouncer. It imports the bouncer's
own FastAPI app (deepguard-bouncer/app/main.py) and adds the ledger's
pages and endpoints to it, so one server on port 8000 serves everything:
every original page and endpoint works exactly as before, plus

  /ledger      register a file on the ledger, verify a file against it
               (the four verdicts), revoke or dispute a record
  /explorer    the block explorer: the whole chain drawn as linked blocks,
               Merkle trees, signatures, witnesses, receipts and a
               tamper lab — every check also redone inside your browser

The only thing done to the original pages is at serve time: one extra
<script> tag (static/bridge.js) is added to the HTML on its way out, which
puts "Ledger" and "Explorer" in the menu and a "provenance" card under the
analysis result. The files on disk are never touched; run the bouncer on
its own (uvicorn main:app) and it is exactly the old app.

How a check works — provenance first, prediction last (blueprint s.4):
  1. exact SHA-256 match to a registered record   -> authentic & unaltered
  2. perceptual-hash or SSCD content match only   -> visually authentic, file modified
  3. no match at all                              -> no provenance: run the AI detector,
                                                     clearly labelled a prediction
  4. match to a record DECLARED AI-generated      -> registered as AI-generated

Start everything (server + two witnesses) with:  python run_deepguard.py
"""

import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

HERE = Path(__file__).resolve().parent
BOUNCER_APP_DIR = HERE.parent / "deepguard-bouncer" / "app"
sys.path.insert(0, str(BOUNCER_APP_DIR))
sys.path.insert(0, str(HERE))

import main as bouncer  # noqa: E402  — the bouncer's app, unchanged
import calibration  # noqa: E402
import content_registry  # noqa: E402
import fingerprint  # noqa: E402

import creators  # noqa: E402
import ledger_core  # noqa: E402

app = bouncer.app

DATA_DIR = Path(os.environ.get("DEEPGUARD_LEDGER_DIR", HERE / "ledger_data"))
STATIC_DIR = HERE / "static"

# 256-bit pHash: unrelated pictures differ in about half their bits (~128),
# and the project's own measurements found crops/rotations land there too
# (116-130). Re-saves and recompressions move only a handful, so 32 of 256
# (12.5%) separates "same picture, different file" from everything else
# with a wide margin on both sides. Crops are SSCD's job, not pHash's.
PHASH_MATCH_MAX_BITS = 32
VIDEO_PHASH_MATCH_MAX_BITS = 32  # mean bits per sampled frame
CONTENT_MATCH_THRESHOLD = content_registry.MATCH_THRESHOLD  # the registry's own 0.4

DECLARED_ORIGINS = {
    "camera": "Original, straight from a camera",
    "edited": "Edited by its creator",
    "ai-generated": "AI-generated, declared by its creator",
}
MAX_TITLE_LENGTH = 200
MAX_REASON_LENGTH = 500


def _witness_config() -> list[dict]:
    raw = os.environ.get("DEEPGUARD_WITNESSES", "Witness A=http://127.0.0.1:8101,Witness B=http://127.0.0.1:8102")
    witnesses = []
    for part in raw.split(","):
        if "=" in part:
            name, url = part.split("=", 1)
            witnesses.append({"name": name.strip(), "url": url.strip().rstrip("/")})
    return witnesses


WITNESSES = _witness_config()

STATE = {"ledger": None, "creators": None}
_INIT_LOCK = threading.Lock()
# Proxies are switched off: the witnesses are on this machine, and a system
# proxy must never sit between the ledger and its witnesses.
_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_ledger() -> ledger_core.Ledger:
    with _INIT_LOCK:
        if STATE["ledger"] is None:
            if bouncer.STATE["signer_private_key"] is None:
                bouncer.load_or_create_signing_key()
            if bouncer.STATE["signer_private_key"] is None:
                raise HTTPException(status_code=503, detail="Deep-Guard's signing key is unavailable.")
            STATE["ledger"] = ledger_core.Ledger(
                DATA_DIR, bouncer.STATE["signer_private_key"], bouncer.STATE["signer_public_key"])
            STATE["creators"] = creators.CreatorBook(DATA_DIR)
            (DATA_DIR / "content_vectors").mkdir(parents=True, exist_ok=True)
    return STATE["ledger"]


def get_creators() -> creators.CreatorBook:
    get_ledger()
    return STATE["creators"]


# ---------------------------------------------------------------------------
# Pages: the two new ones, plus the bridge script on the three old ones
# ---------------------------------------------------------------------------

BRIDGE_TAGS = ('<link rel="stylesheet" href="/ledger-static/ledger.css" />\n'
               '<script src="/ledger-static/bridge.js" defer></script>\n')


def _with_bridge(page_file: str):
    def serve():
        html = (bouncer.STATIC_DIR / page_file).read_text(encoding="utf-8")
        return HTMLResponse(html.replace("</body>", BRIDGE_TAGS + "</body>", 1))
    return serve


# Registered, then moved to the front of the route list so they answer
# before the bouncer's own "/", "/registry" and "/bias-map" handlers.
for _path, _file in (("/", "index.html"), ("/registry", "registry.html"), ("/bias-map", "bias-map.html")):
    app.add_api_route(_path, _with_bridge(_file), methods=["GET"], include_in_schema=False)
    app.router.routes.insert(0, app.router.routes.pop())


@app.get("/ledger", include_in_schema=False)
def ledger_page():
    return FileResponse(str(STATIC_DIR / "ledger.html"))


@app.get("/explorer", include_in_schema=False)
def explorer_page():
    return FileResponse(str(STATIC_DIR / "explorer.html"))


app.mount("/ledger-static", StaticFiles(directory=str(STATIC_DIR)), name="ledger-static")


# ---------------------------------------------------------------------------
# Reading an upload: fingerprints, content vector, detector reading
# ---------------------------------------------------------------------------

def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    bouncer._check_upload_size(file_bytes)
    return file_bytes, Path(file.filename or "upload").name


def _fingerprint_upload(file_bytes: bytes, filename: str) -> dict:
    """The exact and visual fingerprints (Stage 2, reused as-is), plus the
    decoded frames for the detector and the image for SSCD."""
    extension = Path(filename).suffix.lower()
    info = {"file_name": filename[:120], **fingerprint.compute_file_fingerprint(file_bytes)}
    if extension in bouncer.IMAGE_EXTENSIONS:
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Couldn't read that image: {exc}") from exc
        info.update(media_type="image", phash=fingerprint.compute_image_phash(image))
        return {"info": info, "frames": [image], "image": image}
    if extension in bouncer.VIDEO_EXTENSIONS:
        try:
            frames, _, meta = bouncer.sample_frames_from_video(file_bytes, extension)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        info.update(media_type="video", phash_frames=fingerprint.compute_video_phash(frames),
                    video={"resolution": str(meta.get("resolution")),
                           "duration": f"{float(meta.get('duration_seconds') or 0):.2f} s",
                           "frames_sampled": len(frames)})
        return {"info": info, "frames": frames, "image": None}
    supported = ", ".join(sorted(bouncer.IMAGE_EXTENSIONS | bouncer.VIDEO_EXTENSIONS))
    raise HTTPException(status_code=400, detail=f"Unsupported file type '{extension}'. Supported: {supported}")


def _content_vector(image: Optional[Image.Image]) -> Optional[np.ndarray]:
    if image is None or bouncer.STATE["sscd_model"] is None:
        return None
    try:
        return content_registry.compute_content_embedding(bouncer.STATE["sscd_model"], image).astype("<f4")
    except Exception:  # noqa: BLE001 — additive, like everywhere in the bouncer
        bouncer.logger.exception("Couldn't compute the SSCD vector for the ledger")
        return None


def _vector_sha256(vector: np.ndarray) -> str:
    return ledger_core.sha256_hex(np.asarray(vector, dtype="<f4").tobytes())


def _detector_reading(frames: list) -> Optional[dict]:
    """The bouncer's detector, run exactly as /api/analyze runs it."""
    if bouncer.STATE["model"] is None or not frames:
        return None
    probabilities, logits = bouncer.score_frames(frames, with_logits=True)
    if len(frames) == 1:
        raw = probabilities[0]
    else:
        raw = bouncer.aggregate_video_scores(probabilities)["probability_fake"]
    reading = {"probability_fake": round(raw, 4), "calibrated_probability_fake": None,
               "frames_scored": len(frames)}
    if bouncer.STATE["calibration"] is not None and len(frames) == 1:
        reading["calibrated_probability_fake"] = round(calibration.apply_temperature(
            logits[0], bouncer.STATE["calibration"]["detector_temperature"]), 4)
    return reading


def _reading_for_record(reading: Optional[dict]) -> Optional[dict]:
    """Ledger records hold no floats (see ledger_core), so percentages are
    written down as text."""
    if reading is None:
        return None
    out = {"ai_probability_percent": f"{reading['probability_fake'] * 100:.1f}",
           "model": "EfficientNet-B0 detector (deepguard_bouncer.pth)"}
    if reading.get("calibrated_probability_fake") is not None:
        out["calibrated_percent"] = f"{reading['calibrated_probability_fake'] * 100:.1f}"
    return out


# ---------------------------------------------------------------------------
# Witnesses
# ---------------------------------------------------------------------------

def _http_json(url: str, payload=None, timeout: float = 2.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with _HTTP.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except json.JSONDecodeError:
            return exc.code, {}


def _witness_alarm_text(body: dict) -> str:
    detail = body.get("detail")
    if isinstance(detail, dict):
        return detail.get("message", str(detail))
    return str(detail or body)


def sync_witnesses(blocks: Optional[list[dict]] = None) -> list[dict]:
    """Shows every witness the blocks it has not seen yet and collects its
    co-signatures. A witness that already holds a DIFFERENT history is not
    pushed to — its disagreement is the whole point, so it is reported."""
    ledger = get_ledger()
    blocks = ledger.read_blocks() if blocks is None else blocks
    results = []
    for witness in WITNESSES:
        status = {**witness, "online": False, "agrees": None, "height": None, "alarm": None,
                  "first_mismatch": None, "new_cosignatures": 0, "key_id": None}
        try:
            code, body = _http_json(f"{witness['url']}/witness/hashes", timeout=1.5)
            if code != 200:
                raise urllib.error.URLError(f"HTTP {code}")
            status.update(online=True, key_id=body.get("witness_key_id"))
            comparison = ledger_core.compare_with_witness(blocks, body.get("hashes", []))
            status.update(height=comparison["witness_height"], first_mismatch=comparison["first_mismatch"])
            if comparison["first_mismatch"] is not None:
                status.update(agrees=False, alarm=f"Holds a different block #{comparison['first_mismatch']} "
                                                  f"than this ledger shows now.")
            elif any(b.get("_unreadable") for b in blocks):
                status.update(agrees=False, alarm="The ledger file has an unreadable line; nothing was sent.")
            else:
                missing = blocks[comparison["witness_height"]:]
                if missing:
                    code, body = _http_json(f"{witness['url']}/witness/blocks", {"blocks": missing}, timeout=4.0)
                    if code == 200:
                        for cosig in body.get("cosignatures", []):
                            ledger.save_cosignatures(cosig["index"], [cosig])
                        status.update(height=body.get("height"), new_cosignatures=len(missing))
                    else:
                        status.update(agrees=False, alarm=_witness_alarm_text(body))
                if status["alarm"] is None:
                    status["agrees"] = True
        except (urllib.error.URLError, OSError, ValueError):
            status["alarm"] = "Offline — start it with run_deepguard.py to get co-signatures."
        results.append(status)
    return results


def _witness_hashes() -> list[dict]:
    out = []
    for witness in WITNESSES:
        try:
            code, body = _http_json(f"{witness['url']}/witness/hashes", timeout=1.5)
            out.append({**witness, "online": code == 200, "hashes": body.get("hashes", []) if code == 200 else []})
        except (urllib.error.URLError, OSError, ValueError):
            out.append({**witness, "online": False, "hashes": []})
    return out


def _seal_and_sync() -> tuple[Optional[dict], list[dict]]:
    block = get_ledger().seal()
    return block, sync_witnesses()


# ---------------------------------------------------------------------------
# Matching an upload against the ledger
# ---------------------------------------------------------------------------

def _load_vector(record: dict) -> Optional[np.ndarray]:
    path = DATA_DIR / "content_vectors" / f"{record['record_id']}.npy"
    if not path.exists():
        return None
    vector = np.load(path).astype("<f4")
    # The vector file lives off the chain (512 numbers is too big for a
    # block), but its hash is on the chain — a swapped file is ignored.
    return vector if _vector_sha256(vector) == record.get("content_vector_sha256") else None


def _match(ledger, upload: dict, vector: Optional[np.ndarray], blocks: list[dict]) -> list[dict]:
    info = upload["info"]
    candidates = []
    for record, block_index, position in ledger.iter_records(blocks):
        if record.get("type") != "registration":
            continue
        match = {"record": record, "block_index": block_index, "position": position,
                 "kind": None, "phash_bits_different": None, "content_similarity": None}
        if record.get("sha256_merkle_root") == info["sha256_merkle_root"]:
            match["kind"] = "exact"
        if info["media_type"] == "image" and record.get("phash"):
            bits = fingerprint.hamming_distance(info["phash"], record["phash"])
            match["phash_bits_different"] = bits
            if match["kind"] is None and bits <= PHASH_MATCH_MAX_BITS:
                match["kind"] = "visual"
        if info["media_type"] == "video" and record.get("phash_frames"):
            mean_bits = fingerprint.video_similarity(info["phash_frames"], record["phash_frames"])
            match["phash_bits_different"] = round(mean_bits, 1)
            if match["kind"] is None and mean_bits <= VIDEO_PHASH_MATCH_MAX_BITS:
                match["kind"] = "visual"
        if vector is not None and record.get("content_vector_sha256"):
            stored = _load_vector(record)
            if stored is not None:
                similarity = float(np.dot(vector, stored))
                match["content_similarity"] = round(similarity, 4)
                if match["kind"] is None and similarity >= CONTENT_MATCH_THRESHOLD:
                    match["kind"] = "content"
        if match["kind"] is not None:
            candidates.append(match)
    order = {"exact": 0, "visual": 1, "content": 2}
    candidates.sort(key=lambda m: (order[m["kind"]], m["phash_bits_different"] or 0, -(m["content_similarity"] or 0)))
    return candidates


VERDICTS = {
    1: ("authentic_unaltered", "Authentic & unaltered",
        "This exact file is on the ledger, byte for byte. It has not been changed since it was registered."),
    2: ("visually_authentic_modified", "Visually authentic, file modified",
        "This matches a registered original, but it is not the same file: it has been re-saved, "
        "compressed, cropped or edited since. The content is traceable; this exact copy is not the master."),
    3: ("no_provenance", "No provenance found",
        "Nothing on the ledger matches this file. There is no record to prove where it came from, so the "
        "AI detector's reading below is a prediction, not proof."),
    4: ("declared_ai", "Registered as AI-generated",
        "This matches a record whose creator declared it AI-generated. Its origin is known and openly "
        "synthetic — honest AI content, not a disguised fake."),
}


def _describe_match(ledger, match: dict, blocks: list[dict]) -> dict:
    record = match["record"]
    out = {**match, "status": ledger.status_of(record["record_id"], blocks), "sealed": match["block_index"] is not None}
    if out["sealed"]:
        index = match["block_index"]
        out["block_check"] = ledger_core.check_block(blocks[index], index, blocks[index - 1] if index > 0 else None,
                                                     ledger.public_pem)
        out["proof"] = ledger.proof_bundle(record["record_id"], blocks)
        out["proof_ok"] = ledger_core.verify_merkle_proof(out["proof"]["leaf_hash"], out["proof"]["merkle_proof"],
                                                          out["proof"]["block_header"]["merkle_root"])
    return out


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/ledger/summary")
def ledger_summary():
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    good = [b for b in blocks if not b.get("_unreadable")]
    return {
        "block_count": len(blocks),
        "record_count": sum(len(b.get("records") or []) for b in good),
        "pending_count": len(ledger.read_pending()),
        "last_block": ledger_core.block_header(good[-1]) | {"block_hash": good[-1].get("block_hash")} if good else None,
        "service_public_key": ledger.public_pem_str,
        "service_key_id": ledger.key_id,
        "witnesses": WITNESSES,
        "declared_origins": DECLARED_ORIGINS,
        "thresholds": {"phash_bits": PHASH_MATCH_MAX_BITS, "content_similarity": CONTENT_MATCH_THRESHOLD},
        "data_dir": str(DATA_DIR),
    }


@app.get("/api/ledger/chain")
def ledger_chain():
    ledger = get_ledger()
    return {
        "blocks": ledger.read_blocks(),
        "pending": ledger.read_pending(),
        "cosignatures": ledger.read_cosignatures(),
        "service_public_key": ledger.public_pem_str,
        "service_key_id": ledger.key_id,
        "witnesses": WITNESSES,
    }


@app.get("/api/ledger/verify-chain")
def ledger_verify_chain():
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    report = ledger.verify(blocks)
    report["witnesses"] = [
        {"name": w["name"], "url": w["url"], "online": w["online"],
         **(ledger_core.compare_with_witness(blocks, w["hashes"]) if w["online"] else {})}
        for w in _witness_hashes()]
    return report


@app.post("/api/ledger/register")
def ledger_register(file: UploadFile = File(...), title: str = Form(...), creator: str = Form(...),
                    declared_origin: str = Form("camera"), seal_now: bool = Form(True)):
    ledger = get_ledger()
    book = get_creators()
    title = " ".join((title or "").split())[:MAX_TITLE_LENGTH]
    if not title:
        raise HTTPException(status_code=400, detail="Give the file a title.")
    if declared_origin not in DECLARED_ORIGINS:
        raise HTTPException(status_code=400, detail=f"declared_origin must be one of {sorted(DECLARED_ORIGINS)}")
    try:
        identity, created = book.get_or_create(creator)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_bytes, filename = _read_upload(file)
    upload = _fingerprint_upload(file_bytes, filename)
    info = upload["info"]

    # First registration wins. A second person claiming the same exact file
    # has to file a dispute against the first record instead.
    blocks = ledger.read_blocks()
    for record, block_index, _ in ledger.iter_records(blocks):
        if (record.get("type") == "registration" and record.get("sha256_merkle_root") == info["sha256_merkle_root"]
                and not ledger.status_of(record["record_id"], blocks)["revoked"]):
            where = f"block #{block_index}" if block_index is not None else "the pending pool"
            raise HTTPException(status_code=409, detail={
                "message": f"This exact file is already registered as '{record['title']}' by "
                           f"{record.get('creator_handle')} ({where}). If that is wrong, file a dispute against it.",
                "record_id": record["record_id"], "block_index": block_index})

    vector = _content_vector(upload["image"])
    reading = _detector_reading(upload["frames"])
    record = {"type": "registration", "record_id": uuid.uuid4().hex, "title": title,
              "declared_origin": declared_origin, "registered_at": ledger_core.now_iso(), **info}
    if vector is not None:
        record["content_vector_sha256"] = _vector_sha256(vector)
        np.save(DATA_DIR / "content_vectors" / f"{record['record_id']}.npy", vector)
    if reading is not None:
        record["detector_reading"] = _reading_for_record(reading)
    signed = book.sign_record(identity["handle"], record)
    ledger.add_pending(signed)

    block, witnesses = _seal_and_sync() if seal_now else (None, [])
    warning = None
    if reading and declared_origin == "camera" and reading["probability_fake"] >= 0.8:
        warning = ("Declared a camera original, but the AI detector reads it as "
                   f"{reading['probability_fake'] * 100:.0f}% AI-generated. Registered anyway — the ledger "
                   "records the claim and the reading side by side; it does not judge.")
    return {
        "record": signed,
        "creator": identity,
        "creator_created": created,
        "detector": reading,
        "warning": warning,
        "block": ledger_core.block_header(block) | {"block_hash": block["block_hash"]} if block else None,
        "proof": ledger.proof_bundle(signed["record_id"]) if block else None,
        "witnesses": witnesses,
    }


@app.post("/api/ledger/seal")
def ledger_seal():
    block, witnesses = _seal_and_sync()
    return {"block": ledger_core.block_header(block) | {"block_hash": block["block_hash"]} if block else None,
            "message": None if block else "Nothing is waiting to be sealed.", "witnesses": witnesses}


@app.post("/api/ledger/verify")
def ledger_verify(file: UploadFile = File(...), ai_fallback: bool = Form(True)):
    ledger = get_ledger()
    file_bytes, filename = _read_upload(file)
    upload = _fingerprint_upload(file_bytes, filename)
    blocks = ledger.read_blocks()
    vector = _content_vector(upload["image"])
    candidates = [_describe_match(ledger, m, blocks) for m in _match(ledger, upload, vector, blocks)]
    valid = [c for c in candidates if not c["status"]["revoked"]]
    best = valid[0] if valid else None

    if best is None:
        number = 3
    elif best["record"].get("declared_origin") == "ai-generated":
        number = 4
    elif best["kind"] == "exact":
        number = 1
    else:
        number = 2
    key, headline, explanation = VERDICTS[number]
    detector = _detector_reading(upload["frames"]) if number == 3 and ai_fallback else None

    notes = []
    if best and not best["sealed"]:
        notes.append("The matching record is still in the pending pool: it has no block, Merkle proof or "
                     "Deep-Guard block signature yet. Seal it from the explorer.")
    if best and best["sealed"] and not best["block_check"]["ok"]:
        notes.append("WARNING: the block holding this record fails verification. Do not trust this match — "
                     "open the explorer to see what was tampered with.")
    if best and best["status"]["disputes"]:
        notes.append(f"This record has {len(best['status']['disputes'])} open dispute(s) filed against it.")
    revoked_only = [c for c in candidates if c["status"]["revoked"]]
    if number == 3 and revoked_only:
        notes.append("It matches a registration that its creator has since revoked, so that record no longer counts.")

    info = {k: v for k, v in upload["info"].items() if k not in ("phash_frames",)}
    return {
        "verdict": key, "verdict_number": number, "headline": headline, "explanation": explanation,
        "notes": notes, "upload": info, "match": best, "other_matches": [c for c in candidates if c is not best][:5],
        "detector": detector, "thresholds": {"phash_bits": PHASH_MATCH_MAX_BITS,
                                             "content_similarity": CONTENT_MATCH_THRESHOLD},
    }


def _target_registration(ledger, target_record_id: str, blocks):
    record, block_index, _ = ledger.find_record(target_record_id.strip(), blocks)
    if record is None or record.get("type") != "registration":
        raise HTTPException(status_code=404, detail="No registration with that record ID.")
    return record, block_index


@app.post("/api/ledger/revoke")
def ledger_revoke(target_record_id: str = Form(...), creator: str = Form(...), reason: str = Form("")):
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    target, _ = _target_registration(ledger, target_record_id, blocks)
    if creators.slug_of(creator) != target.get("creator_id"):
        raise HTTPException(status_code=403, detail="Only the creator who registered it can revoke it. "
                                                    "Anyone else can file a dispute instead.")
    if ledger.status_of(target["record_id"], blocks)["revoked"]:
        raise HTTPException(status_code=409, detail="That registration is already revoked.")
    record = get_creators().sign_record(creator, {
        "type": "revocation", "record_id": uuid.uuid4().hex, "target_record_id": target["record_id"],
        "reason": " ".join(reason.split())[:MAX_REASON_LENGTH], "filed_at": ledger_core.now_iso()})
    ledger.add_pending(record)
    block, witnesses = _seal_and_sync()
    return {"record": record, "block": ledger_core.block_header(block) | {"block_hash": block["block_hash"]},
            "witnesses": witnesses}


@app.post("/api/ledger/dispute")
def ledger_dispute(target_record_id: str = Form(...), creator: str = Form(...), reason: str = Form(...)):
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    target, _ = _target_registration(ledger, target_record_id, blocks)
    reason = " ".join(reason.split())[:MAX_REASON_LENGTH]
    if len(reason) < 5:
        raise HTTPException(status_code=400, detail="Say why you dispute it (at least a few words).")
    try:
        identity, _ = get_creators().get_or_create(creator)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record = get_creators().sign_record(identity["handle"], {
        "type": "dispute", "record_id": uuid.uuid4().hex, "target_record_id": target["record_id"],
        "reason": reason, "filed_at": ledger_core.now_iso()})
    ledger.add_pending(record)
    block, witnesses = _seal_and_sync()
    return {"record": record, "block": ledger_core.block_header(block) | {"block_hash": block["block_hash"]},
            "witnesses": witnesses}


@app.post("/api/ledger/resolve")
def ledger_resolve(dispute_record_id: str = Form(...), outcome: str = Form(...), note: str = Form("")):
    """Deep-Guard's decision on a dispute, written down like everything else.
    Signed by the block only — the service is the one deciding."""
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    dispute, _, _ = ledger.find_record(dispute_record_id.strip(), blocks)
    if dispute is None or dispute.get("type") != "dispute":
        raise HTTPException(status_code=404, detail="No dispute with that record ID.")
    if outcome not in ("upheld", "rejected"):
        raise HTTPException(status_code=400, detail="outcome must be 'upheld' or 'rejected'.")
    record = {"type": "resolution", "record_id": uuid.uuid4().hex, "target_record_id": dispute["target_record_id"],
              "dispute_record_id": dispute["record_id"], "outcome": outcome,
              "note": " ".join(note.split())[:MAX_REASON_LENGTH], "resolved_by": f"Deep-Guard (key {ledger.key_id})",
              "filed_at": ledger_core.now_iso()}
    ledger.add_pending(record)
    block, witnesses = _seal_and_sync()
    return {"record": record, "block": ledger_core.block_header(block) | {"block_hash": block["block_hash"]},
            "witnesses": witnesses}


@app.get("/api/ledger/record/{record_id}")
def ledger_record(record_id: str):
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    record, block_index, position = ledger.find_record(record_id, blocks)
    if record is None:
        raise HTTPException(status_code=404, detail="No record with that ID.")
    return {"record": record, "block_index": block_index, "position": position,
            "status": ledger.status_of(record_id, blocks), "proof": ledger.proof_bundle(record_id, blocks)}


@app.get("/api/ledger/receipt/{record_id}")
def ledger_receipt(record_id: str):
    ledger = get_ledger()
    bundle = ledger.proof_bundle(record_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="No record with that ID.")
    if not bundle["sealed"]:
        raise HTTPException(status_code=409, detail="This record is not sealed into a block yet.")
    receipt = {
        "receipt_version": 1,
        "issued_at": ledger_core.now_iso(),
        "what_this_is": "Proof that this record sits in a signed block of the Deep-Guard ledger. Check it "
                        "offline with: python tools/verify_ledger.py receipt <this file> <Deep-Guard public key file>",
        **{k: v for k, v in bundle.items() if k != "sealed"},
    }
    return JSONResponse(receipt, headers={
        "Content-Disposition": f'attachment; filename="deepguard-receipt-{record_id[:12]}.json"'})


@app.get("/api/ledger/merkle/{index}")
def ledger_merkle(index: int):
    blocks = get_ledger().read_blocks()
    if not 0 <= index < len(blocks) or blocks[index].get("_unreadable"):
        raise HTTPException(status_code=404, detail="No readable block with that number.")
    records = blocks[index].get("records") or []
    try:
        levels = ledger_core.merkle_levels(records)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"index": index, "levels": levels, "stored_root": blocks[index].get("merkle_root"),
            "leaves": [{"record_id": r.get("record_id"), "type": r.get("type"),
                        "label": r.get("title") or r.get("type")} for r in records]}


@app.get("/api/ledger/creators")
def ledger_creators():
    return {"creators": get_creators().list()}


@app.get("/api/ledger/witnesses")
def ledger_witnesses():
    blocks = get_ledger().read_blocks()
    return {"witnesses": [
        {"name": w["name"], "url": w["url"], "online": w["online"],
         **(ledger_core.compare_with_witness(blocks, w["hashes"]) if w["online"] else {})}
        for w in _witness_hashes()]}


@app.post("/api/ledger/witnesses/sync")
def ledger_witnesses_sync():
    return {"witnesses": sync_witnesses()}


@app.get("/api/ledger/attacks")
def ledger_attacks():
    return {"attacks": ledger_core.ATTACKS}


@app.post("/api/ledger/tamper-lab")
def ledger_tamper_lab(payload: dict = Body(...)):
    """Runs an attack on a COPY of the chain and checks the copy. The real
    files are never written."""
    ledger = get_ledger()
    blocks = ledger.read_blocks()
    try:
        tampered, description = ledger_core.simulate_attack(
            blocks, str(payload.get("attack")), int(payload.get("block_index", -1)), ledger.private_pem)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report = ledger.verify(tampered)
    witnesses = [{"name": w["name"], "online": w["online"],
                  **(ledger_core.compare_with_witness(tampered, w["hashes"]) if w["online"] else {})}
                 for w in _witness_hashes()]
    caught_by_witness = any(w.get("first_mismatch") is not None for w in witnesses)
    if not report["ok"]:
        verdict = f"Caught by the chain itself at block #{report['first_bad_index']}."
    elif caught_by_witness:
        verdict = ("The rewritten chain checks out perfectly on its own — only the witnesses catch it, because "
                   "their copies still hold the original hashes.")
    else:
        verdict = ("The rewritten chain checks out on its own, and no online witness has seen these blocks — start "
                   "the witnesses (run_deepguard.py) and sync them to catch this attack.")
    return {"attack": payload.get("attack"), "description": description, "verdict": verdict,
            "report": report, "witnesses": witnesses, "blocks": tampered}
