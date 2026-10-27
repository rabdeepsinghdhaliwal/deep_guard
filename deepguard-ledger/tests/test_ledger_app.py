"""The ledger wired into the real Deep-Guard app: real models, two real
witness servers on their own ports, a throwaway data folder. Nothing here
touches ledger_data/ or the bouncer's registry files."""
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
