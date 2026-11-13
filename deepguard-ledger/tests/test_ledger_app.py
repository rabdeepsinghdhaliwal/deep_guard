"""The ledger wired into the real Deep-Guard app: real models, two real
witness servers on their own ports, a throwaway data folder. Nothing here
touches ledger_data/ or the bouncer's registry files."""
import asyncio
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import uvicorn
from PIL import Image

LEDGER_DIR = Path(__file__).resolve().parents[1]
REPO = LEDGER_DIR.parent
ART = REPO / "deepguard-bouncer" / "demo_artworks"
TEST_IMAGES = REPO / "test_images"
DALLE = REPO / "generalization_test" / "held_out_images" / "dalle3" / "dalle3_000.webp"
WITNESS_PORTS = (18101, 18102)

sys.path.insert(0, str(LEDGER_DIR))
import ledger_core as lc  # noqa: E402


def _start_witness(name: str, port: int, data_dir: Path):
    os.environ["DEEPGUARD_WITNESS_NAME"] = name
    os.environ["DEEPGUARD_WITNESS_DIR"] = str(data_dir)
    spec = importlib.util.spec_from_file_location(f"witness_{port}", LEDGER_DIR / "witness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = uvicorn.Server(uvicorn.Config(module.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, module


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ledger")
    witnesses = [_start_witness(f"Witness {c}", port, root / f"w{c}") for c, port in zip("AB", WITNESS_PORTS)]
    os.environ["DEEPGUARD_LEDGER_DIR"] = str(root / "data")
    os.environ["DEEPGUARD_CREDENTIALS_DIR"] = str(root / "credentials")
    os.environ["DEEPGUARD_WITNESSES"] = ",".join(f"Witness {c}=http://127.0.0.1:{p}" for c, p in zip("AB", WITNESS_PORTS))
    import ledger_app
    from fastapi.testclient import TestClient
    bouncer = ledger_app.bouncer
    bouncer.CONTENT_REGISTRY_INDEX_PATH = root / "registry.index"
    bouncer.CONTENT_REGISTRY_METADATA_PATH = root / "registry_meta.json"
    bouncer.CONTENT_REGISTRY_FEATURES_DIR = root / "registry_features"
    with TestClient(ledger_app.app) as client:
        yield {"client": client, "app": ledger_app, "root": root, "witnesses": witnesses}
    for server, _ in witnesses:
        server.should_exit = True


def jpg(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def register(client, data: bytes, name: str, title: str, creator: str, origin: str = "camera", seal: bool = True):
    return client.post("/api/ledger/register", files={"file": (name, data)},
                       data={"title": title, "creator": creator, "declared_origin": origin, "seal_now": str(seal).lower()})


def verify(client, data: bytes, name: str, ai: bool = True):
    r = client.post("/api/ledger/verify", files={"file": (name, data)}, data={"ai_fallback": str(ai).lower()})
    assert r.status_code == 200, r.text
    return r.json()


def public_pem(env):
    return env["app"].get_ledger().public_pem


# ------------------------------------------------------------------- pages

def test_new_pages_and_the_bridge_on_old_pages(env):
    c = env["client"]
    assert "Provenance Ledger" in c.get("/ledger").text
    assert "Ledger Explorer" in c.get("/explorer").text
    for path, page in (("/", "index.html"), ("/registry", "registry.html"), ("/bias-map", "bias-map.html")):
        served = c.get(path).text
        original = (env["app"].bouncer.STATIC_DIR / page).read_text(encoding="utf-8")
        assert "/ledger-static/bridge.js" in served and "/ledger-static/bridge.js" not in original
        assert served == original.replace("</body>", env["app"].BRIDGE_TAGS + "</body>", 1)
    assert c.get("/ledger-static/dgcrypto.js").status_code == 200


def test_the_bouncer_still_works_unchanged(env):
    c = env["client"]
    assert c.get("/api/health").json()["model_loaded"]
    r = c.post("/api/analyze", files={"file": ("real2.jpg", (TEST_IMAGES / "real2.jpg").read_bytes())})
    assert r.status_code == 200 and r.json()["verdict"] in ("authentic", "manipulated")
    assert c.post("/api/registry/check", files={"file": ("m.jpg", (ART / "mona_lisa.jpg").read_bytes())}).json()["matches"]


# --------------------------------------------------------- the four verdicts

def test_register_and_the_four_verdicts(env):
    c = env["client"]
    original = (ART / "mona_lisa.jpg").read_bytes()
    r = register(c, original, "mona_lisa.jpg", "Mona Lisa (test)", "Tester")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["block"]["index"] >= 1 and d["record"]["creator_handle"] == "Tester"
    assert lc.verify_receipt(d["proof"], public_pem(env))["ok"]
    assert all(w["agrees"] for w in d["witnesses"]), d["witnesses"]
    assert len(d["proof"]["witness_cosignatures"]) == 2
    assert "detector_reading" in d["record"] and "content_vector_sha256" in d["record"]

    again = register(c, original, "copy.jpg", "Stolen", "Someone Else")
    assert again.status_code == 409

    exact = verify(c, original, "mona_lisa.jpg")
    assert exact["verdict_number"] == 1 and exact["match"]["kind"] == "exact" and exact["detector"] is None
    assert exact["match"]["proof_ok"] and exact["match"]["block_check"]["ok"]

    image = Image.open(io.BytesIO(original)).convert("RGB")
    resaved = verify(c, jpg(image, 60), "whatsapp.jpg")
    assert resaved["verdict_number"] == 2 and resaved["match"]["kind"] == "visual"
    assert resaved["match"]["phash_bits_different"] <= env["app"].PHASH_MATCH_MAX_BITS

    w, h = image.size
    crop = image.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8)))
    cropped = verify(c, jpg(crop), "crop.jpg")
    assert cropped["verdict_number"] == 2 and cropped["match"]["kind"] == "content"
    assert cropped["match"]["content_similarity"] >= 0.4

    unrelated = verify(c, (TEST_IMAGES / "real2.jpg").read_bytes(), "real2.jpg")
    assert unrelated["verdict_number"] == 3 and unrelated["match"] is None
    assert 0 <= unrelated["detector"]["probability_fake"] <= 1


def test_declared_ai_gives_verdict_four(env):
    c = env["client"]
    data = DALLE.read_bytes()
    assert register(c, data, "dalle3_000.webp", "Made with DALL-E 3", "Prompt Artist", "ai-generated").status_code == 200
    d = verify(c, data, "dalle3_000.webp")
    assert d["verdict_number"] == 4 and d["verdict"] == "declared_ai"


# ------------------------------------------------------------- corrections

def test_revoke_dispute_and_resolve(env):
    c = env["client"]
    data = (ART / "great_wave.jpg").read_bytes()
    rid = register(c, data, "great_wave.jpg", "The Great Wave", "Owner").json()["record"]["record_id"]

    assert c.post("/api/ledger/revoke", data={"target_record_id": rid, "creator": "Stranger"}).status_code == 403
    dispute = c.post("/api/ledger/dispute", data={"target_record_id": rid, "creator": "Stranger",
                                                  "reason": "I painted this, not Owner"})
    assert dispute.status_code == 200
    resolve = c.post("/api/ledger/resolve", data={"dispute_record_id": dispute.json()["record"]["record_id"],
                                                  "outcome": "rejected", "note": "Hokusai painted it"})
    assert resolve.status_code == 200
    status = c.get(f"/api/ledger/record/{rid}").json()["status"]
    assert len(status["disputes"]) == 1 and status["resolutions"][0]["outcome"] == "rejected"
    assert verify(c, data, "gw.jpg")["notes"]  # "has 1 open dispute"

    assert c.post("/api/ledger/revoke", data={"target_record_id": rid, "creator": "Owner", "reason": "mistake"}).status_code == 200
    after = verify(c, data, "gw.jpg")
    assert after["verdict_number"] == 3 and any("revoked" in n for n in after["notes"])
    # once revoked, the same file can be registered again
    assert register(c, data, "great_wave.jpg", "The Great Wave", "Hokusai Estate").status_code == 200


def test_pending_records_and_sealing(env):
    c = env["client"]
    data = (ART / "starry_night.jpg").read_bytes()
    d = register(c, data, "starry.jpg", "Starry Night", "Vincent", seal=False).json()
    assert d["block"] is None and d["proof"] is None
    v = verify(c, data, "starry.jpg")
    assert v["verdict_number"] == 1 and not v["match"]["sealed"] and v["notes"]
    sealed = c.post("/api/ledger/seal").json()
    assert sealed["block"]["record_count"] == 1
    assert verify(c, data, "starry.jpg")["match"]["sealed"]


# ------------------------------------------------------------------- video

def _video(path: Path, reupload: bool) -> bytes:
    """A 4-second clip; the "re-upload" is the same clip slightly blurred and
    brightened, as a platform's re-encode would leave it."""
    base = np.array(Image.open(ART / "starry_night.jpg").convert("RGB").resize((320, 240)))[:, :, ::-1]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(40):
        frame = np.roll(base, i * 3, axis=1)
        if reupload:
            frame = cv2.convertScaleAbs(cv2.GaussianBlur(frame, (3, 3), 0), alpha=1.0, beta=8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_video_register_and_verify(env):
    c = env["client"]
    master = _video(env["root"] / "clip.avi", False)
    reencoded = _video(env["root"] / "clip_low.avi", True)
    assert master != reencoded
    d = register(c, master, "clip.avi", "Short clip", "Videographer")
    assert d.status_code == 200, d.text
    assert d.json()["record"]["media_type"] == "video" and d.json()["record"]["phash_frames"]
    assert verify(c, master, "clip.avi")["verdict_number"] == 1
    second = verify(c, reencoded, "clip_low.avi")
    assert second["verdict_number"] == 2 and second["match"]["kind"] == "visual"


# ------------------------------------------------------- proof & checking

def test_receipt_download_and_the_offline_tool(env):
    c = env["client"]
    rid = c.get("/api/ledger/chain").json()["blocks"][1]["records"][0]["record_id"]
    r = c.get(f"/api/ledger/receipt/{rid}")
    assert "attachment" in r.headers["content-disposition"]
    receipt_path = env["root"] / "receipt.json"
    receipt_path.write_text(r.text, encoding="utf-8")
    assert lc.verify_receipt(r.json(), public_pem(env))["ok"]

    key = REPO / "deepguard-bouncer" / "models" / "demo_signer_public.pem"
    tool = LEDGER_DIR / "tools" / "verify_ledger.py"
    chain = env["root"] / "data" / "chain.jsonl"
    assert subprocess.run([sys.executable, str(tool), "receipt", str(receipt_path), str(key)],
                          capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, str(tool), "chain", str(chain), str(key)], capture_output=True).returncode == 0


def test_explorer_endpoints(env):
    c = env["client"]
    chain = c.get("/api/ledger/chain").json()
    assert len(chain["blocks"]) >= 6 and chain["service_key_id"]
    tree = c.get("/api/ledger/merkle/1").json()
    assert tree["levels"][-1][0] == tree["stored_root"]
    assert {x["handle"] for x in c.get("/api/ledger/creators").json()["creators"]} >= {"Tester", "Owner", "Stranger"}
    assert set(c.get("/api/ledger/attacks").json()["attacks"]) == set(lc.ATTACKS)
    report = c.get("/api/ledger/verify-chain").json()
    assert report["ok"] and all(w["agrees"] for w in report["witnesses"])


@pytest.mark.parametrize("attack, caught_by_chain", [
    ("edit_record", True), ("edit_and_rehash", True), ("delete_block", True), ("insider_rewrite", False)])
def test_tamper_lab(env, attack, caught_by_chain):
    c = env["client"]
    d = c.post("/api/ledger/tamper-lab", json={"attack": attack, "block_index": 1}).json()
    assert (not d["report"]["ok"]) == caught_by_chain
    if caught_by_chain:
        assert d["report"]["first_bad_index"] == 1
    else:
        assert all(w["first_mismatch"] == 1 for w in d["witnesses"])
    assert c.get("/api/ledger/verify-chain").json()["ok"]  # the real chain was not touched


def test_a_hand_edit_on_disk_is_caught(env):
    c = env["client"]
    path = env["root"] / "data" / "chain.jsonl"
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    block = json.loads(lines[1])
    block["records"][0]["title"] = "Edited in VS Code"
    lines[1] = json.dumps(block, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        report = c.get("/api/ledger/verify-chain").json()
        assert not report["ok"] and report["first_bad_index"] == 1
    finally:
        path.write_text(original, encoding="utf-8")
    assert c.get("/api/ledger/verify-chain").json()["ok"]


def test_witnesses_catch_a_real_insider_rewrite(env):
    """Deep-Guard rewrites its own file with a perfectly re-signed history.
    The chain alone says "fine"; both witnesses refuse it."""
    c = env["client"]
    app = env["app"]
    ledger = app.get_ledger()
    path = env["root"] / "data" / "chain.jsonl"
    original = path.read_text(encoding="utf-8")
    forged, _ = lc.simulate_attack(ledger.read_blocks(), "insider_rewrite", 1, ledger.private_pem)
    path.write_text("".join(json.dumps(b, sort_keys=True) + "\n" for b in forged), encoding="utf-8")
    try:
        report = c.get("/api/ledger/verify-chain").json()
        assert report["ok"]
        assert all(w["first_mismatch"] == 1 for w in report["witnesses"])
        synced = c.post("/api/ledger/witnesses/sync").json()["witnesses"]
        assert all(w["agrees"] is False and "different block #1" in w["alarm"] for w in synced)
    finally:
        path.write_text(original, encoding="utf-8")
    assert all(w["agrees"] for w in c.post("/api/ledger/witnesses/sync").json()["witnesses"])


# ------------------------------------------------------- Bitcoin anchors

def test_bitcoin_anchors_and_the_collusion_attack(env, monkeypatch):
    """Anchor the newest block (fake calendars and explorers: no network), see
    it pending, confirmed, downloadable and checkable offline, and see it be
    the only thing that catches an insider rewrite the witnesses go along with."""
    import hashlib
    import anchoring
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_bitcoin import FakeBitcoin

    fake = FakeBitcoin()
    monkeypatch.setattr(anchoring, "CALENDARS", ("https://a.example", "https://b.example"))
    monkeypatch.setattr(anchoring, "submit_to_calendar", fake.submit)
    monkeypatch.setattr(anchoring, "fetch_upgrade", fake.fetch)
    monkeypatch.setattr(anchoring, "bitcoin_block", fake.lookup)
    c = env["client"]

    lab = c.post("/api/ledger/tamper-lab", json={"attack": "collusion_rewrite", "block_index": 1}).json()
    assert lab["report"]["ok"] and all(w["colluding"] for w in lab["witnesses"])
    assert lab["anchors"] == [] and "Nothing catches" in lab["verdict"]

    newest = c.get("/api/ledger/anchors").json()["newest_block"]
    r = c.post("/api/ledger/anchor")
    assert r.status_code == 200, r.text
    entry = r.json()["anchor"]
    assert entry["block_index"] == newest and entry["status"] == "pending"
    assert c.post("/api/ledger/anchor").status_code == 409
    listed = next(a for a in c.get("/api/ledger/anchors").json()["anchors"] if a["block_index"] == newest)
    assert listed["chain_agrees"] and listed["paths"][0]["attestation"]["type"] == "pending"

    header = c.get(f"/api/ledger/anchors/{newest}/header")
    assert hashlib.sha256(header.content).hexdigest() == entry["block_hash"]
    assert "attachment" in header.headers["content-disposition"]
    proof = c.get(f"/api/ledger/anchors/{newest}/proof").content
    assert anchoring.load_proof(proof).file_digest.hex() == entry["block_hash"]
    assert c.get("/api/ledger/anchors/999/proof").status_code == 404

    assert c.post("/api/ledger/anchors/upgrade").json()["results"][0]["status"] == "pending"
    fake.mined = True
    up = c.post("/api/ledger/anchors/upgrade?recheck=true").json()
    assert up["results"][0]["status"] == "confirmed", up
    confirmed = next(a for a in up["anchors"] if a["block_index"] == newest)
    assert confirmed["bitcoin"]["height"] == FakeBitcoin.HEIGHT
    assert any(p["attestation"]["type"] == "bitcoin" for p in confirmed["paths"])

    tool = LEDGER_DIR / "tools" / "verify_ledger.py"
    run = subprocess.run([sys.executable, str(tool), "anchors", str(env["root"] / "data"), "--offline"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

    lab = c.post("/api/ledger/tamper-lab", json={"attack": "collusion_rewrite", "block_index": 1}).json()
    assert lab["report"]["ok"] and any(not a["agrees"] for a in lab["anchors"])
    assert "only the anchor" in lab["verdict"]
    assert c.get("/api/ledger/verify-chain").json()["ok"]  # the real chain was never touched


# ------------------------------------------------------- the hidden watermark ID

FIXED_ID = "1" + "0110" * 15  # 61 bits; fixed so the robustness checks below are the same on every run


def _credentialed(d) -> bytes:
    import base64
    return base64.b64decode(d["credentialed_copy"]["base64"])


def test_the_credentialed_copy_carries_a_hidden_id_that_survives_whatsapp(env, monkeypatch):
    import watermark
    import fingerprint
    monkeypatch.setattr(watermark, "new_id", lambda: FIXED_ID)
    c = env["client"]
    original = (TEST_IMAGES / "real4.jpg").read_bytes()
    d = register(c, original, "real4.jpg", "School photo (watermark test)", "Marker").json()
    rec = d["record"]
    assert rec["watermark"] == {"alg": "com.adobe.trustmark.Q", "schema": "BCH_5", "id": watermark.id_hex(FIXED_ID)}
    copy = _credentialed(d)
    assert d["credentialed_copy"]["mime"] == "image/jpeg" and d["credentialed_copy"]["file_name"].endswith(".jpg")
    assert fingerprint.compute_file_fingerprint(copy)["sha256_merkle_root"] == rec["credentialed_copy"]["sha256_merkle_root"]
    assert env["app"].WATERMARKER.read(Image.open(io.BytesIO(copy))) == FIXED_ID
    assert env["app"].WATERMARKER.read(Image.open(io.BytesIO(original))) is None  # the original stays unmarked

    exact = verify(c, copy, d["credentialed_copy"]["file_name"])
    assert exact["verdict_number"] == 1 and exact["match"]["exact_file"] == "credentialed_copy"

    image = Image.open(io.BytesIO(copy)).convert("RGB")
    whatsapp = jpg(image.resize((image.width // 2, image.height // 2)), 70)
    traced = verify(c, whatsapp, "IMG-20261105-WA0001.jpg")
    assert traced["verdict_number"] == 2 and traced["match"]["kind"] == "watermark", traced["match"]
    assert traced["match"]["watermark_id"] == rec["watermark"]["id"] and traced["watermark"]["matched"]

    w, h = image.size
    cropped = jpg(image.crop((int(w * .05), int(h * .05), int(w * .95), int(h * .95))), 80)
    assert verify(c, cropped, "crop.jpg")["match"]["kind"] == "watermark"

    # first registration wins, for copies of the credentialed copy too
    taken = register(c, whatsapp, "mine.jpg", "My photo", "Someone Else")
    assert taken.status_code == 409 and "hidden ID" in taken.json()["detail"]["message"]
    again = register(c, copy, "mine.jpg", "My photo", "Someone Else")
    assert again.status_code == 409 and "credentialed copy" in again.json()["detail"]["message"]


def test_a_png_original_gets_a_png_credentialed_copy(env):
    d = register(env["client"], (TEST_IMAGES / "real1.png").read_bytes(), "real1.png", "Portrait (png)", "Marker").json()
    assert d["credentialed_copy"]["mime"] == "image/png"
    assert _credentialed(d)[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_hidden_id_lifted_onto_another_picture_is_not_trusted(env):
    """The watermark copy attack: someone paints a registered record's ID
    into a different picture. The ID is read, but the content doesn't match
    the record it names, so the ledger refuses to vouch for it."""
    import watermark
    c = env["client"]
    chain = c.get("/api/ledger/chain").json()
    victim = next(r for b in chain["blocks"] for r in b["records"] if r.get("watermark") and "watermark test" in r["title"])
    other = Image.open(REPO / "generalization_test" / "held_out_images" / "dalle3" / "dalle3_002.webp").convert("RGB")
    forged = env["app"].WATERMARKER.embed(other, watermark.bits_from_hex(victim["watermark"]["id"]))
    d = verify(c, jpg(forged, 95), "forged.jpg")
    assert d["watermark"]["id"] == victim["watermark"]["id"]  # the ID really is in the pixels...
    assert not d["watermark"]["matched"] and d["watermark"]["unconfirmed"]  # ...but nothing vouches for it
    assert d["verdict_number"] == 3 and any("copied onto a different picture" in n for n in d["notes"])


# ------------------------------------------------------- Content Credentials (C2PA)

def test_durable_credentials_manifest_then_watermark(env, monkeypatch):
    """The credentialed copy speaks C2PA (signed by this server's demo CA,
    naming its hidden ID and its record). A WhatsApp copy loses that manifest
    but is still traced by the hidden ID: the 'durable' in durable credentials."""
    import watermark
    monkeypatch.setattr(watermark, "new_id", lambda: "0" + "1001" * 15)
    c = env["client"]
    original = (ART / "starry_night.jpg").read_bytes()
    image = Image.open(io.BytesIO(original)).convert("RGB")
    fresh = jpg(image.resize((image.width - 8, image.height - 8)), 93)  # a file no other test registers
    d = register(c, fresh, "night.jpg", "Night sky (credentials test)", "Painter", "edited").json()
    rec, copy = d["record"], _credentialed(d)
    assert rec["credentialed_copy"]["content_credentials"] == "c2pa"

    cr = c.post("/api/credentials/read", files={"file": ("night.credentialed.jpg", copy)}).json()
    assert cr["verdict"] == "ours" and cr["deepguard"]["record_id"] == rec["record_id"]
    assert cr["soft_binding"] == {"alg": "com.adobe.trustmark.Q", "value": "1*" + "0" + "1001" * 15}
    assert cr["evidence"]["layer"] == 0 and cr["evidence"]["status"] == "found"

    exact = verify(c, copy, "night.credentialed.jpg")
    assert exact["verdict_number"] == 1 and exact["credentials"]["verdict"] == "ours"

    small = Image.open(io.BytesIO(copy)).convert("RGB")
    whatsapp = jpg(small.resize((small.width // 2, small.height // 2)), 75)
    traced = verify(c, whatsapp, "IMG-WA0002.jpg")
    assert traced["credentials"]["verdict"] == "none"  # the manifest is gone...
    assert traced["match"]["kind"] == "watermark" and traced["match"]["record"]["record_id"] == rec["record_id"]  # ...the ID is not

    ca = c.get("/api/credentials/ca")
    assert ca.text.startswith("-----BEGIN CERTIFICATE-----") and "attachment" in ca.headers["content-disposition"]


# ------------------------------------------------- the public link (phase 5)

PHONE_A, PHONE_B = "a1" * 16, "b2" * 16


def visitor(browser=None, ip="203.0.113.7"):
    """The headers Cloudflare's tunnel adds to every visitor's request (plus
    the visitor cookie, once the browser has one)."""
    headers = {"cf-connecting-ip": ip, "cf-ray": "8f0c2d1e9a7b4c3d-BOM"}
    if browser:
        headers["cookie"] = f"dg_visitor={browser}"
    return headers


@pytest.fixture
def public(env, monkeypatch):
    """The server as `run_deepguard.py --public` starts it, with fresh limits."""
    app = env["app"]
    monkeypatch.setattr(app.public_mode, "ENABLED", True)
    monkeypatch.setattr(app, "_LIMITER", app.public_mode.RateLimiter())
    monkeypatch.setattr(app, "_HEAVY_JOBS", asyncio.Semaphore(1))
    return app.public_mode


def small_jpg(name: str) -> bytes:
    image = Image.open(TEST_IMAGES / name).convert("RGB")
    image.thumbnail((512, 512))
    return jpg(image, 90)


def test_without_public_mode_the_tunnel_headers_change_nothing(env):
    c = env["client"]
    assert "dgl-public-banner" not in c.get("/ledger", headers=visitor()).text
    assert c.get("/api/ledger/summary", headers=visitor()).json()["data_dir"] is not None
    assert "set-cookie" not in c.get("/explorer", headers=visitor()).headers


def test_visitors_get_the_public_rules_and_the_laptop_does_not(env, public):
    c = env["client"]
    for path in ("/api/ledger/revoke", "/api/ledger/dispute", "/api/ledger/resolve", "/api/ledger/anchor",
                 "/api/ledger/anchors/upgrade", "/api/registry/register"):
        r = c.post(path, headers=visitor(PHONE_A), data={"x": "1"})
        assert r.status_code == 403 and r.json()["detail"] == public.BLOCKED_PATHS[path], path
    assert c.post("/api/ledger/revoke", data={}).status_code == 422  # the laptop reaches the endpoint itself

    page = c.get("/ledger", headers=visitor(PHONE_A)).text
    assert page.index('class="dgl-public-banner"') > page.index("<body")
    assert "dgl-public-banner" in c.get("/", headers=visitor(PHONE_A)).text  # the bouncer's pages too
    assert "dgl-public-banner" not in c.get("/ledger").text

    seen, local = c.get("/api/ledger/summary", headers=visitor(PHONE_A)).json(), c.get("/api/ledger/summary").json()
    assert seen["data_dir"] is None and seen["public_link"] is True
    assert local["data_dir"] == str(env["root"] / "data") and local["public_link"] is False


def test_a_new_visitor_gets_a_cookie_once(env, public):
    c = env["client"]
    cookie = c.get("/explorer", headers=visitor()).headers["set-cookie"]
    flags = cookie.lower()
    assert cookie.startswith("dg_visitor=") and "httponly" in flags and "secure" in flags and "samesite=lax" in flags
    value = cookie.split(";")[0].split("=", 1)[1]
    assert public.browser_of({public.COOKIE: value}) == value
    assert "set-cookie" not in c.get("/explorer", headers=visitor(value)).headers
    assert "set-cookie" not in c.get("/explorer").headers  # never for the laptop


def test_public_uploads_are_capped_even_when_sent_in_chunks(env, public, monkeypatch):
    monkeypatch.setattr(public, "MAX_UPLOAD_BYTES", 50_000)
    c = env["client"]
    big = (TEST_IMAGES / "real3.jpg").read_bytes()
    assert len(big) > 50_000
    r = c.post("/api/ledger/verify", headers=visitor(PHONE_A), files={"file": ("big.jpg", big)})
    assert r.status_code == 413 and "limited to" in r.json()["detail"]
    assert c.post("/api/ledger/verify", files={"file": ("big.jpg", big)}).status_code == 200  # not for the laptop

    as_json = {**visitor(PHONE_A), "content-type": "application/json"}
    chunked = c.post("/api/fingerprint/verify", headers=as_json, content=iter([b" " * 60_000]))
    assert "content-length" not in chunked.request.headers and chunked.status_code == 413
    passed_on = c.post("/api/fingerprint/verify", headers=as_json, content=iter([b'{"no": ', b'"signature"}']))
    assert passed_on.status_code == 400 and "signature" in passed_on.json()["detail"]  # the body arrived whole


def test_public_rate_limits_per_phone_and_per_network(env, public, monkeypatch):
    monkeypatch.setattr(public, "RATES", {"heavy": (2, 60), "write": (1, 60), "any": (300, 60)})
    monkeypatch.setattr(public, "NETWORK_RATES", {"heavy": (3, 60), "write": (2, 60), "any": (3000, 60)})
    c = env["client"]
    picture = jpg(Image.new("RGB", (64, 64), (200, 120, 40)))

    def read(headers):
        return c.post("/api/credentials/read", headers=headers, files={"file": ("p.jpg", picture)})

    assert [read(visitor(PHONE_A)).status_code for _ in range(3)] == [200, 200, 429]  # phone A's share
    assert [read(visitor(PHONE_B)).status_code for _ in range(2)] == [200, 429]       # the Wi-Fi's share
    assert read(visitor(PHONE_B, ip="198.51.100.9")).status_code == 200              # another network
    assert all(read({}).status_code == 200 for _ in range(5))                        # the laptop: no limits
    refused = read(visitor(PHONE_A))
    assert refused.headers["retry-after"] == "30" and "Wait a moment" in refused.json()["detail"]


def test_heavy_checks_take_turns(env, public, monkeypatch):
    app, c = env["app"], env["client"]
    picture = jpg(Image.new("RGB", (64, 64), (20, 120, 240)))
    monkeypatch.setattr(public, "HEAVY_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(app, "_HEAVY_JOBS", asyncio.Semaphore(0))  # someone else's check is running
    busy = c.post("/api/credentials/read", headers=visitor(PHONE_A), files={"file": ("p.jpg", picture)})
    assert busy.status_code == 503 and busy.headers["retry-after"] == "10"
    assert c.get("/api/ledger/summary", headers=visitor(PHONE_A)).status_code == 200  # light requests don't queue

    turn = asyncio.Semaphore(1)
    monkeypatch.setattr(app, "_HEAVY_JOBS", turn)
    assert c.post("/api/credentials/read", files={"file": ("p.jpg", picture)}).status_code == 200
    assert c.post("/api/credentials/read", headers=visitor(PHONE_A), files={"file": ("p.jpg", picture)}).status_code == 200
    assert not turn.locked()  # every turn was given back


def test_public_creator_names_stay_with_their_owner(env, public):
    c = env["client"]

    def register_as(headers, data, creator):
        return c.post("/api/ledger/register", headers=headers, files={"file": ("p.jpg", data)},
                      data={"title": "From my phone", "creator": creator, "declared_origin": "camera"})

    assert register(c, small_jpg("real3.jpg"), "r3.jpg", "Laptop photo", "Laptop Owner").status_code == 200
    taken = register_as(visitor(PHONE_A), small_jpg("ai3.png"), "laptop owner")
    assert taken.status_code == 403 and "belongs to someone on this laptop" in taken.json()["detail"]

    mine = register_as(visitor(PHONE_A), small_jpg("ai3.png"), "Visitor Priya")
    assert mine.status_code == 200, mine.text
    who = mine.json()["creator"]
    assert who["public_visitor"] is True and who["public_browser"] == public.browser_tag(PHONE_A)
    assert PHONE_A not in json.dumps(c.get("/api/ledger/creators").json())  # only a hash of the cookie is kept

    other = register_as(visitor(PHONE_B), small_jpg("ai4.png"), "Visitor Priya")
    assert other.status_code == 403 and "Another visitor" in other.json()["detail"]
    again = register_as(visitor(PHONE_A), small_jpg("ai4.png"), "visitor priya")
    assert again.status_code == 200, again.text
    assert again.json()["record"]["creator_key_id"] == mine.json()["record"]["creator_key_id"]


def test_visitors_are_given_the_witnesses_public_addresses(env, public, monkeypatch):
    monkeypatch.setenv("DEEPGUARD_WITNESS_PUBLIC_URLS", "Witness A=https://quiet-river-stone-a.trycloudflare.com/")
    c = env["client"]
    seen = {w["name"]: w["url"] for w in c.get("/api/ledger/chain", headers=visitor(PHONE_A)).json()["witnesses"]}
    assert seen == {"Witness A": "https://quiet-river-stone-a.trycloudflare.com",
                    "Witness B": f"http://127.0.0.1:{WITNESS_PORTS[1]}"}  # B has no public address: unchanged
    assert c.get("/api/ledger/summary", headers=visitor(PHONE_A)).json()["witnesses"][0]["url"].startswith("https://")
    local = {w["name"]: w["url"] for w in c.get("/api/ledger/chain").json()["witnesses"]}
    assert local["Witness A"] == f"http://127.0.0.1:{WITNESS_PORTS[0]}"


def test_a_witness_takes_blocks_only_from_its_own_laptop(env, public):
    import httpx
    base = f"http://127.0.0.1:{WITNESS_PORTS[0]}"
    with httpx.Client(trust_env=False, timeout=10) as http:
        before = http.get(f"{base}/witness/info").json()
        forged = {"blocks": [{"index": 0, "block_hash": "f" * 64}]}
        r = http.post(f"{base}/witness/blocks", json=forged, headers=visitor())
        assert r.status_code == 403 and "not over the public link" in r.json()["detail"]
        after = http.get(f"{base}/witness/info").json()
        assert after["alarms"] == before["alarms"] and after["height"] == before["height"]  # no false alarm
        assert http.post(f"{base}/witness/blocks", json={"blocks": []}).status_code == 200  # the ledger's own posts


def test_the_share_page_and_the_public_status(env, public, monkeypatch):
    c = env["client"]
    monkeypatch.setattr(public, "PUBLIC_URL", "https://quiet-river-stone-a.trycloudflare.com")
    status = c.get("/api/public/status").json()
    assert status["public"] is True and status["url"] == "https://quiet-river-stone-a.trycloudflare.com"
    assert status["qr_svg"].startswith("<svg") and status["limits"]["upload_mb"] == 15
    svg_tag = status["qr_svg"].split(">", 1)[0]
    assert "viewBox" in svg_tag and "width=" not in svg_tag  # scales down to a phone instead of being cut off
    assert "/api/ledger/revoke" in status["limits"]["owner_only"]
    assert "Scan to use Deep-Guard" in c.get("/share").text
    monkeypatch.setattr(public, "PUBLIC_URL", "")
    assert c.get("/api/public/status").json()["qr_svg"] is None
