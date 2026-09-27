"""
Deep-Guard -- before/after snapshots of the running app.

Every step on the roadmap changes the app. This tool records what the app
does *now* for a fixed set of inputs, so that after a change the two
recordings can be compared and nothing changes by accident.

    # 1. start the server (from deepguard-bouncer/app):  uvicorn main:app
    # 2. from deepguard-bouncer/:
    python evaluation/snapshot.py capture 2026-09-27_baseline
    #    ...make a change, restart the server...
    python evaluation/snapshot.py capture 2026-09-28_after_step0
    python evaluation/snapshot.py compare 2026-09-27_baseline 2026-09-28_after_step0

What one capture writes to evaluation/snapshots/<label>/:
  api.json      the server's full answers for the canary inputs, with every
                embedded picture replaced by its size and SHA-256 (small,
                but a changed heatmap still shows up)
  summary.json  the numbers a person reads off the page, per input
  screens/      full-page screenshots of what a person sees (needs
                playwright and Chrome; skipped with a note otherwise)
  inputs/       the two generated inputs (see below)
screens/ and inputs/ are gitignored: they are for looking at, and are
rebuilt by the next capture anyway.

The canary inputs are files in this repository, plus two made from them:
a 4-second video whose first half is a real photo and second half a
DALL-E 3 image (exercises the video path and the "part of this video" rule), and
a cropped, mirrored copy of the Mona Lisa (exercises "why it matched").
Nothing is ever registered -- the tool only reads from the registry.
"""

import argparse
import base64
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageChops, ImageOps

BOUNCER = Path(__file__).resolve().parents[1]
REPO = BOUNCER.parent
SNAPSHOTS = BOUNCER / "evaluation" / "snapshots"

CANARY_IMAGES = {
    "real1": REPO / "test_images" / "real1.png",
    "real3": REPO / "test_images" / "real3.jpg",
    "ai1": REPO / "test_images" / "ai1.png",
    "ai3": REPO / "test_images" / "ai3.png",
    "dalle3_000": REPO / "generalization_test" / "held_out_images" / "dalle3" / "dalle3_000.webp",
    "midjourney_002": REPO / "generalization_test" / "held_out_images" / "midjourney" / "midjourney-v5_002.webp",
    "mona_lisa": BOUNCER / "demo_artworks" / "mona_lisa.jpg",
}
# Which canaries the page screenshots cover (every canary is in api.json).
SCREENSHOT_ANALYSES = ["real3", "ai1", "dalle3_000", "canary_video"]
REGISTRY_CHECKS = ["mona_lisa", "mona_copy", "real3"]

CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".webp": "image/webp", ".mp4": "video/mp4"}

# How far a number may move between two captures before `compare` calls it
# a change. Everything here is deterministic and should match almost
# exactly. The Monte Carlo Dropout readings are not: they use fresh random
# dropout masks on every call and swing by several points on borderline
# images, so they are shown for information and never counted as a change
# (RANDOM_KEYS) -- regressions show up in the deterministic numbers.
TOLERANCES = {
    "probability_fake": 0.005,
    "mean_probability": 0.005,
    "peak_probability": 0.005,
    "flagged_ratio": 0.0,
    "attribution_top_p": 0.01,
    "frequency_anomaly": 0.01,
    "similarity": 0.005,
}
RANDOM_KEYS = {"consistency_mean", "consistency_std"}
# Fields that differ on every call by design (a new timestamp, therefore a
# new signature) or measure the machine rather than the app.
VOLATILE_KEYS = {"timestamp_utc", "signature", "elapsed_seconds", "generated_at_utc"}


# ------------------------------------------------------------------ inputs --

def make_canary_video(path: Path) -> None:
    """32 frames at 8 fps: 16 of a real photo, then 16 of an AI image the
    detector does catch -- so the "part of this video" rule has work to do."""
    import cv2
    import numpy as np

    size = (640, 480)
    frames = []
    for key in ("real3", "dalle3_000"):
        rgb = Image.open(CANARY_IMAGES[key]).convert("RGB").resize(size, Image.BICUBIC)
        frames += [cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)] * 16
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def make_mona_copy(path: Path) -> None:
    """The registry's own Mona Lisa, cropped to its middle and mirrored."""
    im = Image.open(CANARY_IMAGES["mona_lisa"]).convert("RGB")
    w, h = im.size
    copy = ImageOps.mirror(im.crop((int(w * 0.20), int(h * 0.10), int(w * 0.85), int(h * 0.80))))
    copy.save(path, "JPEG", quality=90)


# --------------------------------------------------------------- recording --

def scrub(value):
    """Replaces embedded pictures (base64 strings) with their size + hash."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, str) and k.endswith("_base64"):
                raw = base64.b64decode(v)
                out[k] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            else:
                out[k] = scrub(v)
        return out
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def page_verdict(d: dict) -> str:
    """The headline the analysis page shows -- mirrors render() in static/script.js."""
    pct = d["probability_fake"] * 100
    if (d.get("input_type") == "video" and d.get("flagged_ratio", 0) >= 0.12
            and d.get("peak_probability", 0) >= 0.75 and pct < 80):
        return "part of this video looks AI-generated"
    if pct >= 80:
        return "looks AI-generated"
    if pct <= 20:
        return "looks authentic"
    return "inconclusive"


def summarise_analysis(d: dict) -> dict:
    ex = d.get("explainability") or {}
    mc = ex.get("model_consistency") or {}
    freq = ex.get("frequency_analysis") or {}
    att = d.get("generator_attribution") or {}
    probs = att.get("probabilities") or {}
    top = next(iter(probs.items()), (None, None))
    fp = d.get("fingerprint") or {}
    s = {
        "input_type": d.get("input_type"),
        "probability_fake": d.get("probability_fake"),
        "server_verdict": d.get("verdict"),
        "page_verdict": page_verdict(d),
        "attribution_top": top[0],
        "attribution_top_p": top[1],
        "attribution_answer": att.get("answer"),  # added in Step 0; absent before
        "consistency_mean": mc.get("mean_probability"),
        "consistency_std": mc.get("std_probability"),
        "frequency_anomaly": freq.get("anomaly_score"),
        "phash": fp.get("phash"),
        "sha256_merkle_root": fp.get("sha256_merkle_root"),
    }
    if d.get("input_type") == "video":
        s.update({k: d.get(k) for k in ("mean_probability", "peak_probability", "frames_analysed",
                                        "frames_flagged", "flagged_ratio")})
    return s


def summarise_check(d: dict) -> dict:
    out = []
    for m in d.get("matches", []):
        ex = m.get("explanation") or {}
        g = ex.get("geometry") or {}
        out.append({
            "title": m.get("title"),
            "similarity": m.get("similarity"),
            "shared_points": ex.get("shared_points"),
            "strength": ex.get("strength"),
            "mirrored": g.get("mirrored"),
            "original_area_shown": g.get("original_area_shown"),
        })
    return {"matches": out}


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("branch", "--show-current"),
            "uncommitted_files": len(run("status", "--porcelain").splitlines())}


def registry_titles() -> list:
    path = BOUNCER / "models" / "content_registry_metadata.json"
    if not path.exists():
        return []
    return [r.get("title") for r in json.loads(path.read_text(encoding="utf-8"))]


def capture(label: str, base_url: str, screens: bool) -> None:
    out = SNAPSHOTS / label
    if out.exists():
        raise SystemExit(f"{out} already exists -- pick a new label.")
    (out / "inputs").mkdir(parents=True)

    inputs = dict(CANARY_IMAGES)
    inputs["canary_video"] = out / "inputs" / "canary_video.mp4"
    make_canary_video(inputs["canary_video"])
    inputs["mona_copy"] = out / "inputs" / "mona_copy.jpg"
    make_mona_copy(inputs["mona_copy"])

    client = httpx.Client(base_url=base_url, timeout=600)
    health = client.get("/api/health").json()
    if not health.get("model_loaded"):
        raise SystemExit(f"Server at {base_url} has no model loaded: {health}")

    def post(route: str, key: str) -> dict:
        path = inputs[key]
        files = {"file": (path.name, path.read_bytes(), CONTENT_TYPES[path.suffix.lower()])}
        started = time.perf_counter()
        r = client.post(route, files=files)
        r.raise_for_status()
        print(f"  {route:<22} {key:<16} {time.perf_counter() - started:5.1f} s")
        return r.json()

    api = {"health": health, "analyze": {}, "registry_check": {}}
    summary = {"analyze": {}, "registry_check": {}}
    for key in [*CANARY_IMAGES, "canary_video"]:
        d = post("/api/analyze", key)
        api["analyze"][key] = scrub(d)
        summary["analyze"][key] = summarise_analysis(d)
    for key in REGISTRY_CHECKS:
        d = post("/api/registry/check", key)
        api["registry_check"][key] = scrub(d)
        summary["registry_check"][key] = summarise_check(d)
    bias = client.get("/api/bias-map").json()
    api["bias_map"] = scrub(bias)
    summary["bias_map"] = {"total_images": bias.get("total_images"),
                           "overall_accuracy": bias.get("overall_accuracy")}

    meta = {
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "git": git_state(),
        "python": platform.python_version(),
        "registry_titles": registry_titles(),
        "inputs": {k: {"path": str(p.relative_to(REPO) if p.is_relative_to(REPO) else p),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in inputs.items()},
    }
    (out / "api.json").write_text(json.dumps({"meta": meta, **api}, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=1), encoding="utf-8")

    if screens:
        take_screenshots(base_url, inputs, out / "screens")
    print(f"Wrote {out}")


def take_screenshots(base_url: str, inputs: dict, folder: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [screens skipped] playwright is not installed (pip install -r evaluation/requirements-eval.txt)")
        return
    folder.mkdir(parents=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 900},
                                   reduced_motion="reduce").new_page()

        def shot(name: str) -> None:
            page.wait_for_timeout(1200)  # let the gauge and number settle
            page.screenshot(path=str(folder / f"{name}.jpg"), full_page=True, type="jpeg", quality=80)
            print(f"  screenshot {name}")

        page.goto(base_url + "/")
        page.wait_for_function("document.getElementById('statusText').textContent.trim() !== 'connecting…'")
        shot("01_home")
        for i, key in enumerate(SCREENSHOT_ANALYSES, start=2):
            page.goto(base_url + "/")
            page.set_input_files("#fileInput", str(inputs[key]))
            page.click("#runBtn")
            page.wait_for_selector("#result:not([hidden])", timeout=300_000)
            shot(f"{i:02d}_analyze_{key}")
        page.goto(base_url + "/registry")
        shot("06_registry")
        page.set_input_files("#checkFile", str(inputs["mona_copy"]))
        page.click("#checkBtn")
        page.wait_for_selector("#checkResult:not([hidden])", timeout=300_000)
        shot("07_registry_check_mona_copy")
        page.goto(base_url + "/bias-map")
        page.wait_for_load_state("networkidle")
        shot("08_bias_map")
        browser.close()


# ---------------------------------------------------------------- compare --

def compare_values(key: str, a, b) -> str:
    if a == b:
        return "same"
    if key in RANDOM_KEYS:
        return "random by design"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and key in TOLERANCES:
        return "within tolerance" if abs(a - b) <= TOLERANCES[key] else "CHANGED"
    return "CHANGED"


def value_diffs(a, b, path=""):
    """Every leaf present in both trees whose value differs: (path, a, b,
    verdict). Hashes and strings must match exactly; numbers use TOLERANCES
    by key name (or MC_TOLERANCE under model_consistency), else exactly.
    Keys present on one side only are reported separately (key_paths).
    Monte Carlo Dropout readings (under model_consistency) are random by
    design and come back as "random by design", never as a change."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) & set(b)):
            if k not in VOLATILE_KEYS:
                yield from value_diffs(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path, f"{len(a)} items", f"{len(b)} items", "CHANGED"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from value_diffs(x, y, f"{path}[{i}]")
    elif a != b:
        if ".model_consistency." in path:
            yield path, a, b, "random by design"
            return
        key = path.rsplit(".", 1)[-1].split("[")[0]
        numbers = isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
        tol = TOLERANCES.get(key)
        ok = numbers and tol is not None and abs(a - b) <= tol
        yield path, a, b, "within tolerance" if ok else "CHANGED"


def key_paths(value, prefix=""):
    """Every key path in a JSON tree, list positions collapsed to []."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in VOLATILE_KEYS:
                continue
            yield f"{prefix}.{k}"
            yield from key_paths(v, f"{prefix}.{k}")
    elif isinstance(value, list):
        for v in value:
            yield from key_paths(v, prefix + "[]")


def compare(label_a: str, label_b: str) -> int:
    a_dir, b_dir = SNAPSHOTS / label_a, SNAPSHOTS / label_b
    sa = json.loads((a_dir / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((b_dir / "summary.json").read_text(encoding="utf-8"))
    lines, changed = [], 0

    for section in ("analyze", "registry_check"):
        for key in sorted(set(sa[section]) | set(sb[section])):
            va, vb = sa[section].get(key), sb[section].get(key)
            if va is None or vb is None:
                lines.append(f"{section}/{key}: only in {'B' if va is None else 'A'}")
                changed += 1
                continue
            if section == "registry_check":
                va = {f"match{i}.{k}": v for i, m in enumerate(va["matches"]) for k, v in m.items()} | {"match_count": len(va["matches"])}
                vb = {f"match{i}.{k}": v for i, m in enumerate(vb["matches"]) for k, v in m.items()} | {"match_count": len(vb["matches"])}
            for field in sorted(set(va) | set(vb)):
                verdict = compare_values(field.split(".")[-1], va.get(field), vb.get(field))
                if verdict != "same":
                    lines.append(f"{section}/{key}.{field}: {va.get(field)!r} -> {vb.get(field)!r}  [{verdict}]")
                    changed += verdict == "CHANGED"

    api_a = json.loads((a_dir / "api.json").read_text(encoding="utf-8"))
    api_b = json.loads((b_dir / "api.json").read_text(encoding="utf-8"))
    for section in ("analyze", "registry_check"):
        paths_a = set(key_paths(api_a[section]))
        paths_b = set(key_paths(api_b[section]))
        for p in sorted(paths_b - paths_a):
            lines.append(f"new field    {section}{p}")
        for p in sorted(paths_a - paths_b):
            lines.append(f"removed field {section}{p}")
            changed += 1
        # Values too, not just which fields exist: a changed heatmap hash or
        # a moved probability anywhere in the answer must show up here.
        for path, va, vb, verdict in value_diffs(api_a[section], api_b[section]):
            if verdict == "CHANGED":
                lines.append(f"value        {section}{path}: {va!r} -> {vb!r}  [CHANGED]")
                changed += 1
    bias = [d for d in value_diffs(api_a.get("bias_map"), api_b.get("bias_map")) if d[3] == "CHANGED"]
    if bias:  # expected whenever the Bias Map is regenerated; one line, not hundreds
        lines.append(f"bias map: {len(bias)} value(s) differ (expected only when the exam was re-scored)")

    report_dir = SNAPSHOTS / f"compare_{label_a}__{label_b}"
    report_dir.mkdir(exist_ok=True)
    for shot_a in sorted((a_dir / "screens").glob("*.jpg")) if (a_dir / "screens").exists() else []:
        shot_b = b_dir / "screens" / shot_a.name
        if not shot_b.exists():
            lines.append(f"screen {shot_a.name}: missing in B")
            continue
        ia, ib = Image.open(shot_a).convert("RGB"), Image.open(shot_b).convert("RGB")
        note = "" if ia.size == ib.size else f" (size {ia.size} -> {ib.size})"
        w, h = min(ia.width, ib.width), min(ia.height, ib.height)
        diff = ImageChops.difference(ia.crop((0, 0, w, h)), ib.crop((0, 0, w, h))).convert("L")
        mask = diff.point(lambda v: 255 if v > 40 else 0)
        pct = 100 * sum(mask.histogram()[255:]) / (w * h)
        lines.append(f"screen {shot_a.name}: {pct:.2f}% of pixels differ{note}")
        if pct > 0.5 or note:
            overlay = ib.crop((0, 0, w, h)).copy()
            overlay.paste((255, 0, 0), mask=mask)
            overlay.save(report_dir / f"diff_{shot_a.name}", quality=80)

    report = "\n".join(lines) or "No differences."
    (report_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n{changed} change(s) outside tolerance. Report and diff images: {report_dir}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="record the running app")
    cap.add_argument("label")
    cap.add_argument("--base-url", default="http://127.0.0.1:8000")
    cap.add_argument("--no-screens", action="store_true", help="API answers only")
    cmp_ = sub.add_parser("compare", help="compare two captures")
    cmp_.add_argument("label_a")
    cmp_.add_argument("label_b")
    args = parser.parse_args()

    if args.command == "capture":
        capture(args.label, args.base_url, screens=not args.no_screens)
    else:
        sys.exit(1 if compare(args.label_a, args.label_b) else 0)


if __name__ == "__main__":
    main()
