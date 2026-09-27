"""
Deep-Guard -- sits the exam: scores every exam-set image with the app as it
is, and writes the report card.

    python evaluation/run_exam.py 2026-09-27_baseline                 # score + report
    python evaluation/run_exam.py 2026-09-27_baseline --fit           # also fit and save the calibration
    python evaluation/run_exam.py 2026-09-27_baseline --write-bias-map   # also refresh the Bias Map page

Everything goes through the code the server runs. The app is imported
in-process (as the test suite does), with the content registry redirected to
a temporary folder so the real registry is never touched, and every exam
image is sent through POST /api/analyze exactly as the page sends it. To
measure what the page does not show, the detector's raw score (its logit)
and the source panel's full answer are also computed for every image --
including AI images the detector wrongly calls real.

Writes evaluation/results/<label>/:
  per_image.csv   one row per image: label, what the page said, the numbers behind it
  summary.json    every metric below, machine-readable
  report.md       the same, readable (the baseline's copy is evaluation/BASELINE.md)
  *.png           reliability diagram, results by generator and by real-photo source

The calibration set is scored too, but only to *fit* two numbers -- the
detector's temperature and the source panel's "unknown" threshold -- which
are then judged on the exam set they never saw.
"""

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BOUNCER = Path(__file__).resolve().parents[1]
APP = BOUNCER / "app"
EVAL = BOUNCER / "evaluation"
DATA = Path(os.environ.get("DEEPGUARD_DATA", Path.home() / "deepguard-data"))
SET_DIRS = {"exam": DATA / "exam_set", "calibration": DATA / "calibration_set"}
MANIFESTS = {"exam": EVAL / "exam_set_manifest.csv", "calibration": EVAL / "calibration_set_manifest.csv"}
CALIBRATION_FILE = BOUNCER / "models" / "detector_calibration.json"
BIAS_MAP_RESULTS = BOUNCER / "bias_map" / "results.json"

AI_BAR, REAL_BAR = 0.80, 0.20   # the analysis page's verdict thresholds (static/script.js)
MIN_GROUP = 20                  # smaller groups are shown but flagged "too small to trust"

sys.path.insert(0, str(APP))


# ------------------------------------------------------------------ helpers --

def page_verdict(p: float) -> str:
    """What the analysis page says for an image with P(AI) = p."""
    return "ai" if p >= AI_BAR else ("real" if p <= REAL_BAR else "inconclusive")


def wilson(k: int, n: int, z: float = 1.96):
    """95% interval for a proportion k/n -- honest about small groups where a
    plain k/n would look more certain than it is."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def auc_interval(labels, scores, reps: int = 1000, seed: int = 0):
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(reps):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() != labels[idx].max():
            vals.append(auc(labels[idx], scores[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def rate(k, n):
    return round(k / n, 4) if n else None


def long_side_bucket(r) -> str:
    s = max(int(r["width"] or 0), int(r["height"] or 0))
    return ("under 512 px" if s < 512 else "512-1023 px" if s < 1024 else
            "1024-2047 px" if s < 2048 else "2048 px and over")


REAL_SOURCE_NAMES = {
    "openfake-core-test": lambda r: "ImageNet photos" if "imagenet" in r["credit"] else "DOCCI photos",
    "openfake-reddit-test": lambda r: "Reddit photography posts",
    "cnndetection": lambda r: "CNNDetection real set (LSUN, ImageNet, faces)",
    "genimage": lambda r: "GenImage real set (ImageNet)",
    "wikimedia-commons": lambda r: "Wikimedia Commons (India)",
    "deepguard-original-test-images": lambda r: "project's original test photos",
}


def user_supplied(r) -> bool:
    """Pictures the team adds through incoming/ (phones, WhatsApp, ChatGPT,
    Gemini). Reported in their own section, so adding them never moves the
    headline that every roadmap step is compared against."""
    return (r.get("source") or "").startswith("user-")


def detector_overlap(r) -> bool:
    """True when the image may not be new to the detector (see overlap_risk in
    the manifest): shown in the group tables, kept out of the headline."""
    return "detector:" in (r.get("overlap_risk") or "")


def group_label(r) -> str:
    """One line per generator, and one per kind of real photo. Groups that may
    not be new to the detector carry a tag saying why."""
    risk = r.get("overlap_risk") or ""
    if "FFHQ" in risk:
        name = "real: FFHQ faces" if r["label"] == "real" else "stylegan faces (whichfaceisreal)"
        return name + " [in the detector's training data]"
    if "demo images" in risk:
        return ("real: project's original test photos" if r["label"] == "real"
                else "project's original AI images") + " [used during development]"
    if user_supplied(r):
        return {"user-phone_real": "your phone photos", "user-whatsapp_real": "your photos after WhatsApp",
                "user-whatsapp_ai": "AI images after WhatsApp", "user-chatgpt_ai": "your ChatGPT images",
                "user-gemini_ai": "your Gemini images"}.get(r["source"], r["source"])
    if r["label"] == "real":
        return "real: " + REAL_SOURCE_NAMES.get(r["source"], lambda _: r["source"])(r)
    if r["source"] == "openfake-reddit-test":
        return "AI-subreddit posts (tool not recorded)"
    if r["source"] == "deepguard-original-test-images":
        return "project's original AI images (tool not recorded)"
    g = r["generator"]
    return {"midjourney_imagenet": "midjourney-5 (GenImage)", "sdv4_imagenet": "sd-1.4 (GenImage)",
            "sdv5_imagenet": "sd-1.5 (GenImage)", "glide_imagenet": "glide", "adm_imagenet": "adm",
            "vqdm_imagenet": "vqdm", "wukong_imagenet": "wukong", "biggan_imagenet": "biggan (GenImage)",
            "whichfaceisreal": "stylegan faces (whichfaceisreal)"}.get(g, g)


SOURCE_NAMES = {
    "openfake-core-test": "OpenFake test split: generators no training set contains",
    "openfake-reddit-test": "OpenFake Reddit set: pictures as they circulate online",
    "cnndetection": "CNNDetection: older GANs (2017-2020)",
    "genimage": "GenImage: older diffusion models and GANs",
    "deepguard-generalization-test": "Our 44 outside images (DALL-E 3, MidJourney v5)",
    "wikimedia-commons": "Wikimedia Commons pictures",
    "deepguard-original-test-images": "Project's original 8 test images",
}


# ---------------------------------------------------------------- scoring --

def load_app():
    os.chdir(APP)
    import main
    tmp = Path(tempfile.mkdtemp(prefix="deepguard-exam-"))
    main.CONTENT_REGISTRY_INDEX_PATH = tmp / "registry.index"
    main.CONTENT_REGISTRY_METADATA_PATH = tmp / "registry_metadata.json"
    main.CONTENT_REGISTRY_FEATURES_DIR = tmp / "features"
    return main


def raw_scores(main, image):
    """The detector's logit and the source panel's 8 logits, computed with the
    exact models and preprocessing the server holds."""
    import torch
    import attribution
    with torch.no_grad():
        z = float(main.STATE["model"](main.STATE["preprocess"](image).unsqueeze(0).to(main.STATE["device"])).item())
        att = main.STATE["attribution"]
        a_logits = att["model"](attribution.PREPROCESS(image).unsqueeze(0).to(main.STATE["device"]))[0].tolist()
    return z, a_logits, att["class_order"]


PROBE_SIDE = 256


def size_probe(main) -> list[dict]:
    """A controlled test the exam's make-up can't answer by itself: small
    images in the exam are mostly from *older* generators, so "the detector
    misses small images" and "the detector misses older generators" are
    tangled. Here the same modern AI images and the same large real photos
    (OpenFake core/test: its generators and its DOCCI photos) are shrunk so
    the long side is 256 px -- the size of the older generators' images --
    and scored again. Nothing else about the pictures changes."""
    import calibration
    with open(MANIFESTS["exam"], newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["purpose"] == "detector" and r["source"] == "openfake-core-test"
                and (r["label"] == "ai" or "docci" in r["credit"])]
    print(f"size probe: {len(rows)} images shrunk to {PROBE_SIDE} px", flush=True)
    out = []
    for r in rows:
        small = Image.open(SET_DIRS["exam"] / r["file"]).convert("RGB")
        small.thumbnail((PROBE_SIDE, PROBE_SIDE), Image.LANCZOS)
        z, _, _ = raw_scores(main, small)
        p = round(calibration.sigmoid(z), 4)
        out.append({"id": f"{r['id']}-{PROBE_SIDE}px", "set": "exam", "purpose": "probe", "label": r["label"],
                    "source": r["source"], "generator": r["generator"], "credit": r["credit"],
                    "variant_of": r["id"], "transform": f"shrunk to {PROBE_SIDE} px long side (Lanczos)",
                    "logit": z, "p_ai": calibration.sigmoid(z), "p_ai_page": p, "verdict": page_verdict(p)})
    return out


def run_probe_only() -> list[dict]:
    from fastapi.testclient import TestClient
    main = load_app()
    with TestClient(main.app):
        return size_probe(main)


def score_all(label: str) -> list[dict]:
    from fastapi.testclient import TestClient
    import calibration

    main = load_app()
    out = []
    with TestClient(main.app) as client:
        for set_name in ("exam", "calibration"):
            with open(MANIFESTS[set_name], newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            todo = [r for r in rows if r["file"].lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
            print(f"{set_name}: scoring {len(todo)} images", flush=True)
            for n, r in enumerate(todo, 1):
                path = SET_DIRS[set_name] / r["file"]
                data = path.read_bytes()
                image = Image.open(io.BytesIO(data)).convert("RGB")   # exactly as /api/analyze reads it
                z, a_logits, classes = raw_scores(main, image)
                rec = {k: r.get(k, "") for k in ("id", "set", "purpose", "label", "source", "generator",
                                                 "generator_group", "attribution_class", "content", "subject",
                                                 "width", "height", "format", "variant_of", "transform",
                                                 "expected", "credit", "overlap_risk")}
                rec["logit"] = z
                rec["p_ai"] = calibration.sigmoid(z)
                m = max(a_logits)
                exps = [math.exp(x - m) for x in a_logits]
                probs = [e / sum(exps) for e in exps]
                top = int(np.argmax(probs))
                rec.update({"attr_top": classes[top], "attr_top_p": probs[top],
                            "attr_energy": m + math.log(sum(exps)),  # log-sum-exp of the logits
                            "attr_probs": dict(zip(classes, [round(p, 5) for p in probs]))})
                if set_name == "exam" and r["purpose"] == "detector":
                    t0 = time.perf_counter()
                    resp = client.post("/api/analyze", files={"file": (path.name, data)})
                    rec["seconds"] = round(time.perf_counter() - t0, 3)
                    resp.raise_for_status()
                    d = resp.json()
                    rec["p_ai_page"] = d["probability_fake"]
                    if abs(d["probability_fake"] - rec["p_ai"]) > 6e-5:
                        raise SystemExit(f"{r['id']}: API said {d['probability_fake']}, direct score {rec['p_ai']:.6f}"
                                         " -- the harness is not scoring what the server scores")
                    mc = (d.get("explainability") or {}).get("model_consistency") or {}
                    rec["wobble"] = mc.get("std_probability")
                    rec["page_shows_source_panel"] = page_verdict(d["probability_fake"]) == "ai" and \
                        d.get("generator_attribution") is not None
                else:
                    rec["p_ai_page"] = round(rec["p_ai"], 4)
                rec["verdict"] = page_verdict(rec["p_ai_page"])
                out.append(rec)
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
        out += size_probe(main)
    return out


def detector_evidence(rec: dict, temperature: float) -> dict:
    """The detector's answer for one image in the trust report's common
    format (app/evidence.py) -- written for every exam image, so the format
    is exercised (and validated) on real answers from Step 0 on."""
    import calibration
    from evidence import Evidence, PointsTo, Status, Strength
    p_cal = calibration.apply_temperature(rec["logit"], temperature)
    v = rec["verdict"]
    points = {"ai": PointsTo.AI, "real": PointsTo.REAL, "inconclusive": PointsTo.NEITHER}[v]
    confidence = p_cal if v == "ai" else (1 - p_cal if v == "real" else max(p_cal, 1 - p_cal))
    # No number in the sentence: the page's raw percentage and the calibrated
    # one differ a lot (T = 10.6), and `confidence` already holds the honest one.
    claim = {"ai": "Our detector reads this as AI-generated.",
             "real": "Our detector reads this as a real photo.",
             "inconclusive": "Our detector can't tell whether this is AI-generated."}[v]
    return Evidence(
        layer=2, check="efficientnet-b0", title="Our AI-image detector", status=Status.FOUND,
        points_to=points, strength=Strength.INDICATION, claim=claim, confidence=round(confidence, 4),
        details={"p_ai": rec["p_ai_page"], "p_ai_calibrated": round(p_cal, 4), "temperature": temperature},
        limits=["A statistical guess from one model, not proof.",
                "Weak on generators it was not trained on (see evaluation/BASELINE.md)."],
    ).validate().to_dict()


# ---------------------------------------------------------------- metrics --

def outcome_counts(rows):
    ai = [r for r in rows if r["label"] == "ai"]
    real = [r for r in rows if r["label"] == "real"]
    c = {
        "n": len(rows), "n_ai": len(ai), "n_real": len(real),
        "ai_called_ai": sum(r["verdict"] == "ai" for r in ai),
        "ai_called_real": sum(r["verdict"] == "real" for r in ai),
        "ai_inconclusive": sum(r["verdict"] == "inconclusive" for r in ai),
        "real_called_real": sum(r["verdict"] == "real" for r in real),
        "real_called_ai": sum(r["verdict"] == "ai" for r in real),
        "real_inconclusive": sum(r["verdict"] == "inconclusive" for r in real),
    }
    correct = c["ai_called_ai"] + c["real_called_real"]
    wrong = c["ai_called_real"] + c["real_called_ai"]
    c.update({
        "caught": rate(c["ai_called_ai"], len(ai)), "missed": rate(c["ai_called_real"], len(ai)),
        "cleared": rate(c["real_called_real"], len(real)), "false_alarm": rate(c["real_called_ai"], len(real)),
        "correct": rate(correct, len(rows)), "wrong": rate(wrong, len(rows)),
        "inconclusive": rate(c["ai_inconclusive"] + c["real_inconclusive"], len(rows)),
        "accuracy_at_50": rate(sum((r["p_ai"] >= 0.5) == (r["label"] == "ai") for r in rows), len(rows)),
        "trustworthy": len(rows) >= MIN_GROUP,
    })
    if ai and real and len(rows) >= 10:
        c["auc"] = round(auc([r["label"] == "ai" for r in rows], [r["logit"] for r in rows]), 4)
    return c


def grouped(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return {k: outcome_counts(v) for k, v in sorted(groups.items())}


def calibration_section(cal_rows, exam_rows):
    import calibration as cal
    from sklearn.linear_model import LogisticRegression
    fit_rows = [r for r in cal_rows if r["label"] in ("ai", "real")]
    y_fit = [int(r["label"] == "ai") for r in fit_rows]
    t = cal.fit_temperature([r["logit"] for r in fit_rows], y_fit)
    # For comparison only: Platt scaling, which adds a bias term (a*z + b) and so
    # can also correct a model that leans one way overall. Not used by the app.
    platt = LogisticRegression(C=1e6).fit([[r["logit"]] for r in fit_rows], y_fit)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])

    def evaluate(rows, to_p):
        y = [int(r["label"] == "ai") for r in rows]
        p = [to_p(r["logit"]) for r in rows]
        wrong = sum((page_verdict(pi) == "ai" and yi == 0) or (page_verdict(pi) == "real" and yi == 1)
                    for pi, yi in zip(p, y))
        unsure = sum(page_verdict(pi) == "inconclusive" for pi in p)
        nll = -sum(math.log(min(max(pi if yi else 1 - pi, 1e-12), 1.0)) for pi, yi in zip(p, y)) / len(p)
        return {"ece": round(cal.expected_calibration_error(p, y), 4), "brier": round(cal.brier_score(p, y), 4),
                "nll": round(nll, 4), "confident_wrong_verdicts": wrong, "inconclusive_verdicts": unsure,
                "n": len(rows), "reliability": cal.reliability_table(p, y)}

    ways = {"raw": lambda z: cal.sigmoid(z),
            "calibrated": lambda z: cal.apply_temperature(z, t),
            "platt": lambda z: cal.sigmoid(a * z + b)}
    return {"temperature": round(t, 4), "platt": {"a": round(a, 4), "b": round(b, 4)},
            "fitted_on": f"calibration set, {len(fit_rows)} images",
            "calibration_set": {k: evaluate(fit_rows, f) for k, f in ways.items()},
            "exam_set": {k: evaluate(exam_rows, f) for k, f in ways.items()}}


MAX_UNKNOWN_NAMED = 0.05


def choose_unknown_threshold(known: list, unknown: list) -> tuple[float, str]:
    """The source panel's naming rule, written as a policy rather than taken
    from a fragile optimum (a Youden-style best balance flipped between "never
    name" and "always name" on resamples of the calibration set):
      - if its top probability does not separate known from unknown tools on
        the calibration set (AUROC at or below 0.5), naming is switched off --
        threshold 1.0, and the app names a tool only when its top probability
        is *above* the threshold, so 1.0 means never;
      - otherwise, the lowest threshold that keeps unknown-tool images named
        at or below MAX_UNKNOWN_NAMED of the time on the calibration set."""
    a = auc([1] * len(known) + [0] * len(unknown), [r["attr_top_p"] for r in known + unknown])
    if a is None or a <= 0.5:
        return 1.0, (f"naming switched off: on the calibration set its top probability separates known from "
                     f"unknown tools no better than chance (AUROC {a:.3f})")
    for t in sorted({r["attr_top_p"] for r in unknown}) + [1.0]:
        if sum(r["attr_top_p"] > t for r in unknown) / len(unknown) <= MAX_UNKNOWN_NAMED:
            t = math.ceil(t * 10000) / 10000   # round up: rounding down could let that image be named
            return t, (f"names a tool only above {100 * t:.1f}%, the lowest bar that keeps unknown-tool "
                                 f"images named at most {100 * MAX_UNKNOWN_NAMED:.0f}% of the time (AUROC {a:.3f})")
    return 1.0, "naming switched off"


def attribution_section(cal_rows, exam_rows):
    """How well the source panel separates its 8 known tools from everything
    else, and the "unknown" threshold that follows (fitted on calibration)."""
    def split(rows):
        # "Unknown" must mean *known to be outside* the 8 tools. Images whose
        # tool was never recorded (Reddit posts, the project's original AI
        # images, Commons faces of unstated StyleGAN version) might come from
        # one of the 8, so they are left out of this test entirely.
        ai = [r for r in rows if r["label"] == "ai"]
        known = [r for r in ai if r["attribution_class"]]
        unknown = [r for r in ai if not r["attribution_class"] and r["generator"] not in ("", "unknown")
                   and "not stated" not in r["generator"]]
        return known, unknown

    known_c, unknown_c = split(cal_rows)
    known_e, unknown_e = split(exam_rows)
    res = {"known_classes_available": sorted({r["attribution_class"] for r in known_c + known_e}),
           "calibration": {"known": len(known_c), "unknown": len(unknown_c)},
           "exam": {"known": len(known_e), "unknown": len(unknown_e)}}
    for name, key in (("top_probability", "attr_top_p"), ("energy", "attr_energy")):
        res[f"auroc_{name}"] = {
            "calibration": round(auc([1] * len(known_c) + [0] * len(unknown_c), [r[key] for r in known_c + unknown_c]), 4),
            "exam": round(auc([1] * len(known_e) + [0] * len(unknown_e), [r[key] for r in known_e + unknown_e]), 4)
            if known_e and unknown_e else None}
    best, policy = choose_unknown_threshold(known_c, unknown_c)
    res["unknown_threshold"], res["policy"] = best, policy

    def judge(known, unknown, t):
        # The app names a tool only when its top probability is ABOVE t.
        k_named_right = sum(r["attr_top_p"] > t and r["attr_top"] == r["attribution_class"] for r in known)
        k_named_wrong = sum(r["attr_top_p"] > t and r["attr_top"] != r["attribution_class"] for r in known)
        return {"known_named_correctly": rate(k_named_right, len(known)),
                "known_named_wrongly": rate(k_named_wrong, len(known)),
                "known_said_unknown": rate(sum(r["attr_top_p"] <= t for r in known), len(known)),
                "unknown_said_unknown": rate(sum(r["attr_top_p"] <= t for r in unknown), len(unknown)),
                "unknown_named_a_tool": rate(sum(r["attr_top_p"] > t for r in unknown), len(unknown)),
                "n_known": len(known), "n_unknown": len(unknown)}

    res["before_unknown_answer"] = {"calibration": judge(known_c, unknown_c, -1.0), "exam": judge(known_e, unknown_e, -1.0)}
    res["with_unknown_answer"] = {"calibration": judge(known_c, unknown_c, best), "exam": judge(known_e, unknown_e, best)}
    res["top1_by_known_class_exam"] = {
        c: {"n": len(v), "named_correctly": rate(sum(r["attr_top"] == c for r in v), len(v)),
            "mean_top_probability": round(statistics.mean(r["attr_top_p"] for r in v), 4)}
        for c, v in sorted(defaultdict(list, {c: [r for r in known_e if r["attribution_class"] == c]
                                              for c in {r["attribution_class"] for r in known_e}}).items())}
    res["mean_top_probability_unknown_exam"] = round(statistics.mean(r["attr_top_p"] for r in unknown_e), 4) if unknown_e else None
    return res


def _rounded(x, digits=4):
    return None if x is None else round(x, digits)


def robustness_section(by_id, rows):
    pairs = [(by_id[r["variant_of"]], r) for r in rows if r["purpose"] == "robustness" and r["variant_of"] in by_id]
    if not pairs:
        return None
    moved = [abs(b["p_ai"] - a["p_ai"]) for a, b in pairs]
    flips = Counter(f"{a['verdict']} -> {b['verdict']}" for a, b in pairs if a["verdict"] != b["verdict"])
    ai_pairs = [(a, b) for a, b in pairs if a["label"] == "ai"]
    return {
        "pairs": len(pairs),
        "mean_change_in_p_ai": round(statistics.mean(moved), 4),
        "median_change_in_p_ai": round(statistics.median(moved), 4),
        "verdict_changed": sum(a["verdict"] != b["verdict"] for a, b in pairs),
        "verdict_changes": dict(flips),
        # Pairs can all be one kind (e.g. only phone photos and their WhatsApp
        # copies): then there is no AI average and no AUC, not a crash.
        "ai_images_mean_p_ai": {"original": round(statistics.mean(a["p_ai"] for a, _ in ai_pairs), 4),
                                "social_copy": round(statistics.mean(b["p_ai"] for _, b in ai_pairs), 4)}
        if ai_pairs else None,
        "auc": {"originals": _rounded(auc([a["label"] == "ai" for a, _ in pairs], [a["logit"] for a, _ in pairs])),
                "social_copies": _rounded(auc([b["label"] == "ai" for _, b in pairs], [b["logit"] for _, b in pairs]))},
        "originals": outcome_counts([a for a, _ in pairs]),
        "social_copies": outcome_counts([b for _, b in pairs]),
    }


def size_probe_section(by_id, probes):
    pairs = [(by_id[p["variant_of"]], p) for p in probes if p["variant_of"] in by_id]
    if not pairs:
        return None
    ai = [(a, b) for a, b in pairs if a["label"] == "ai"]
    real = [(a, b) for a, b in pairs if a["label"] == "real"]
    return {
        "side_px": PROBE_SIDE, "ai_images": len(ai), "real_photos": len(real),
        "ai_called_ai": {"full_size": rate(sum(a["verdict"] == "ai" for a, _ in ai), len(ai)),
                         "shrunk": rate(sum(b["verdict"] == "ai" for _, b in ai), len(ai))},
        "ai_called_real": {"full_size": rate(sum(a["verdict"] == "real" for a, _ in ai), len(ai)),
                           "shrunk": rate(sum(b["verdict"] == "real" for _, b in ai), len(ai))},
        "real_called_ai": {"full_size": rate(sum(a["verdict"] == "ai" for a, _ in real), len(real)),
                           "shrunk": rate(sum(b["verdict"] == "ai" for _, b in real), len(real))},
        "real_called_real": {"full_size": rate(sum(a["verdict"] == "real" for a, _ in real), len(real)),
                             "shrunk": rate(sum(b["verdict"] == "real" for _, b in real), len(real))},
        "auc": {"full_size": round(auc([a["label"] == "ai" for a, _ in pairs], [a["logit"] for a, _ in pairs]), 4),
                "shrunk": round(auc([b["label"] == "ai" for _, b in pairs], [b["logit"] for _, b in pairs]), 4)},
        "mean_p_ai_ai_images": {"full_size": round(statistics.mean(a["p_ai"] for a, _ in ai), 4),
                                "shrunk": round(statistics.mean(b["p_ai"] for _, b in ai), 4)},
    }


def summarise(records):
    exam = [r for r in records if r["set"] == "exam"]
    det_all = [r for r in exam if r["purpose"] == "detector" and r["label"] in ("ai", "real")]
    det = [r for r in det_all if not detector_overlap(r) and not user_supplied(r)]   # the headline
    user = [r for r in det_all if user_supplied(r)]
    cal_rows = [r for r in records if r["set"] == "calibration"]
    by_id = {r["id"]: r for r in exam}
    y = [r["label"] == "ai" for r in det]
    overall = outcome_counts(det)
    overall["auc_interval_95"] = auc_interval(y, [r["logit"] for r in det])
    for k, n_key, num in (("caught", "n_ai", "ai_called_ai"), ("false_alarm", "n_real", "real_called_ai"),
                          ("missed", "n_ai", "ai_called_real"), ("cleared", "n_real", "real_called_real")):
        overall[f"{k}_interval_95"] = wilson(overall[num], overall[n_key])
    secs = [r["seconds"] for r in det if r.get("seconds")]
    return {
        "overall": overall,
        "by_source": grouped(det, lambda r: SOURCE_NAMES.get(r["source"], r["source"])),
        "by_generator": grouped([r for r in det_all if not user_supplied(r)], group_label),
        "user_supplied": grouped(user, group_label) if user else {},
        "user_supplied_count": len(user),
        "by_kind": grouped(det, lambda r: r["generator_group"]),
        "by_subject": grouped(det, lambda r: r["subject"] or "unlabelled"),
        "by_size": grouped(det, long_side_bucket),
        "by_format": grouped(det, lambda r: r["format"]),
        "by_content": grouped(det, lambda r: r["content"]),
        "calibration": calibration_section(cal_rows, det),
        "attribution": attribution_section(cal_rows, det),
        "robustness_social_copies": robustness_section(
            by_id, [r for r in exam if not (r.get("transform") or "").startswith("whatsapp")]),
        "robustness_whatsapp": robustness_section(
            by_id, [r for r in exam if (r.get("transform") or "").startswith("whatsapp")]),
        "size_probe": size_probe_section(by_id, [r for r in exam if r["purpose"] == "probe"]),
        "layer0_files_detector_opinion": {r["id"]: {"p_ai": round(r["p_ai"], 4), "verdict": r["verdict"],
                                                    "expected_c2pa": json.loads(r["expected"] or "{}").get("c2pa")}
                                          for r in exam if r["purpose"] == "layer0"},
        "speed_seconds_per_image": {"median": round(statistics.median(secs), 3),
                                    "p90": round(float(np.percentile(secs, 90)), 3)} if secs else None,
        "left_out_of_headline": {"n": sum(detector_overlap(r) for r in det_all),
                                 "why": sorted({r["overlap_risk"] for r in det_all if detector_overlap(r)})},
        "counts": {"exam_scored": len(exam), "exam_detector_images": len(det),
                   "calibration_scored": len(cal_rows),
                   "by_purpose": dict(Counter(r["purpose"] for r in exam))},
    }


# ------------------------------------------------------------------ charts --

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]   # categorical slots 1-2 (validated: CVD dE 24.7, contrast >= 3:1)


def _axes(fig_w, fig_h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    return fig, ax, plt


def chart_reliability(cal_section, out: Path):
    fig, ax, plt = _axes(6.2, 5.2)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.90, "perfectly honest", color=MUTED, ha="right", fontsize=9, rotation=40)
    for i, (key, name) in enumerate((("raw", "as the page shows it"),
                                     ("calibrated", f"after temperature scaling (T = {cal_section['temperature']:.2f})"))):
        tab = [b for b in cal_section["exam_set"][key]["reliability"] if b["count"]]
        xs, ys = [b["mean_predicted"] for b in tab], [b["fraction_ai"] for b in tab]
        sizes = [max(30, min(260, 6 * b["count"])) for b in tab]
        ax.plot(xs, ys, color=SERIES[i], lw=2, zorder=2, label=name)
        ax.scatter(xs, ys, s=sizes, color=SERIES[i], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("what the detector said (average P(AI) in the group)", color=INK2)
    ax.set_ylabel("how many of those images really were AI", color=INK2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Does the detector's percentage mean what it says?", color=INK, loc="left", fontsize=12, pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, "Exam set, detector images. Dot size = number of images in the group.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_bars(groups: dict, value_key: str, title: str, xlabel: str, out: Path, note: str):
    """Horizontal bars, one hue, sorted. Groups under 5 images are left out of
    the picture (they are in the tables); groups under MIN_GROUP get a dagger."""
    items = sorted(((k, v) for k, v in groups.items() if v.get(value_key) is not None and v["n"] >= 5),
                   key=lambda kv: kv[1][value_key])
    fig, ax, plt = _axes(8.6, 0.3 * len(items) + 1.9)
    names = [f"{k}  (n={v['n']})" for k, v in items]
    vals = [v[value_key] * 100 for _, v in items]
    ax.barh(names, vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for i, (val, (_, v)) in enumerate(zip(vals, items)):
        ax.text(val + 1.2, i, f"{val:.0f}%" + ("" if v["trustworthy"] else " †"),
                va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlabel, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, note + f"  † fewer than {MIN_GROUP} images.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ report --

def pct(x, digits=0):
    """Rounded half-up, exactly as the Bias Map page's Math.round does, so the
    report and the page never disagree on a number (Python's own rounding is
    half-to-even: 12.5 -> 12, where the page shows 13)."""
    if x is None:
        return "—"
    scale = 10 ** digits
    return f"{math.floor(100 * x * scale + 0.5) / scale:.{digits}f}%"


def fmt3(x):
    return "—" if x is None else f"{x:.3f}"


def interval(iv):
    return "" if not iv or iv[0] is None else f" (95% range {100 * iv[0]:.0f}–{100 * iv[1]:.0f}%)"


def group_table(groups: dict, kind: str) -> str:
    lines = ["| Group | n | Called AI-generated | Inconclusive | Called authentic |", "|---|---:|---:|---:|---:|"]
    for k, v in sorted(groups.items(), key=lambda kv: (-(kv[1]["n_ai"] > 0), kv[0])):
        if v["n_ai"] and not v["n_real"]:
            cells = (pct(v["caught"]), pct(rate(v["ai_inconclusive"], v["n_ai"])), pct(v["missed"]))
        elif v["n_real"] and not v["n_ai"]:
            cells = (pct(v["false_alarm"]), pct(rate(v["real_inconclusive"], v["n_real"])), pct(v["cleared"]))
        else:
            cells = (f"AI {pct(v['caught'])} / real {pct(v['false_alarm'])}", pct(v["inconclusive"]),
                     f"AI {pct(v['missed'])} / real {pct(v['cleared'])}")
        flag = "" if v["trustworthy"] else " †"
        lines.append(f"| {k}{flag} | {v['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def panel_reading(a: dict) -> str:
    """The source panel's conclusion, following the rule actually in force."""
    ba, wa = a["before_unknown_answer"]["exam"], a["with_unknown_answer"]["exam"]
    if a["unknown_threshold"] >= 1.0:
        return ("An AUROC at or below 0.5 is no better than a coin toss: the panel is *more* sure of itself on "
                "tools it has never seen than on the ones it was trained on, so no threshold on its confidence "
                "can tell the two apart. Naming is therefore switched off: the panel answers \"can't tell which "
                "tool made this\" for every image, instead of giving a confidently wrong name to every image "
                f"from an unknown tool. The price is the {pct(ba['known_named_correctly'])} of known-tool images it "
                "used to name correctly, and even that figure is an upper bound, because its training set "
                "(ArtiFact) may contain some of those images. A useful answer needs a better source model "
                "(roadmap Step 4), not a better threshold.")
    return (f"Its confidence does separate known from unknown tools, so it names a tool above the bar: on the "
            f"exam it named {pct(wa['known_named_correctly'])} of known-tool images correctly and "
            f"{pct(wa['known_named_wrongly'])} wrongly, and named a tool for {pct(wa['unknown_named_a_tool'])} of "
            "images from tools it does not know.")


def size_reading(sp: dict) -> str:
    """The size test's conclusion, worded from its own numbers -- and only as
    far as the test can see (it changes the file's size, which the detector's
    own 256x256 resize mostly erases anyway)."""
    drop = sp["ai_called_ai"]["full_size"] - sp["ai_called_ai"]["shrunk"]
    if abs(drop) < 0.10:
        return (f"The file's size makes little difference ({pct(sp['ai_called_ai']['full_size'])} → "
                f"{pct(sp['ai_called_ai']['shrunk'])} caught), as expected from the detector's own resize. This does "
                "not explain *why* it misses older generators: their images differ from the modern ones in more "
                "than size (made at 256 px natively, mostly lossless PNG, different subjects), and this exam "
                "cannot separate those. Training on many generators (roadmap Step 4) addresses all of them.")
    return (f"Size matters. Shrinking modern AI images to {sp['side_px']} px changes how often they are caught "
            f"({pct(sp['ai_called_ai']['full_size'])} → {pct(sp['ai_called_ai']['shrunk'])}), so part of what "
            "looked like a weakness on older generators is a weakness on small images; training should include "
            "images at many sizes.")


def write_report(label: str, s: dict, out: Path, meta: dict) -> str:
    o, c, a, rb = s["overall"], s["calibration"], s["attribution"], s["robustness_social_copies"]
    ce, cc, cp = c["exam_set"]["raw"], c["exam_set"]["calibrated"], c["exam_set"]["platt"]
    naming_rule = ("naming switched off" if a["unknown_threshold"] >= 1.0
                   else f"names a tool only above {100 * a['unknown_threshold']:.1f}%")
    first_bin = next(b for b in ce["reliability"] if b["count"])
    wa, ba = a["with_unknown_answer"]["exam"], a["before_unknown_answer"]["exam"]
    md = f"""# Deep-Guard exam results: {label}

Measured {meta['date']} on `{meta['commit']}` ({meta['branch']}; "-dirty" means uncommitted changes were
present), with the detector (`deepguard_bouncer.pth`) and source panel (`deepguard_attribution.pth`)
exactly as the server runs them. The {o['n'] + s['left_out_of_headline']['n'] + s['user_supplied_count']} detector images went through
`POST /api/analyze`, the same call the analysis page makes; the derived files (social-media copies,
watermarked files, the size test) were scored directly with the same model objects, which were checked
to give the API's answer to 4 decimal places on every one of those detector images.

## The exam

{o['n']} labelled images the detector has never been tuned on ({o['n_ai']} AI-generated, {o['n_real']} real),
plus {s['counts']['by_purpose'].get('robustness', 0)} social-media-style copies, {s['counts']['by_purpose'].get('layer0', 0)} Content
Credentials test images and {s['counts']['by_purpose'].get('layer1', 0)} watermarked files that later steps use.
Where every image came from, with its licence and SHA-256: `evaluation/exam_set_manifest.csv`.
What the groups mean: `evaluation/README.md`.

{s['left_out_of_headline']['n']} further images are in the tables below but **not** in the headline, because they
may not be new to the detector ({'; '.join(w.split(': ', 1)[1] for w in s['left_out_of_headline']['why'])}).

The page gives one of three answers: **"looks AI-generated"** (P(AI) ≥ 80%), **"inconclusive"**
(20–80%), or **"looks authentic"** (≤ 20%). The numbers below count those answers.

## Headline

| | Result |
|---|---|
| AI images the page calls AI-generated | **{pct(o['caught'])}**{interval(o['caught_interval_95'])} |
| AI images the page calls **authentic** (confident miss) | **{pct(o['missed'])}**{interval(o['missed_interval_95'])} |
| Real photos the page calls authentic | **{pct(o['cleared'])}**{interval(o['cleared_interval_95'])} |
| Real photos the page calls **AI-generated** (false alarm) | **{pct(o['false_alarm'])}**{interval(o['false_alarm_interval_95'])} |
| Inconclusive (either kind) | {pct(o['inconclusive'])} |
| Ranking quality, AUC (1.0 = perfect, 0.5 = coin toss) | {o['auc']:.3f} (95% range {o['auc_interval_95'][0]:.3f}–{o['auc_interval_95'][1]:.3f}) |
| Accuracy if the page simply split at 50% | {pct(o['accuracy_at_50'])} |
| Time per image (median / slowest 10%) | {s['speed_seconds_per_image']['median']:.2f} s / {s['speed_seconds_per_image']['p90']:.2f} s |

![Results by generator](by_generator.png)

![Real photos by source](real_sources.png)

## By where the images came from

{group_table(s['by_source'], 'source')}

## By generator (and by kind of real photo)

{group_table(s['by_generator'], 'generator')}

† fewer than {MIN_GROUP} images: shown, but too few to trust on their own.

## By kind of generator

{group_table(s['by_kind'], 'kind')}

## By image size, file format, content

{group_table(s['by_size'], 'size')}

{group_table(s['by_format'], 'format')}

{group_table(s['by_content'], 'content')}

## By subject (machine-labelled by CLIP, so approximate)

{group_table(s['by_subject'], 'subject')}

## Does "90%" mean right 9 times in 10? (calibration)

Temperature scaling was fitted on the separate calibration set ({c['fitted_on']}) and judged here.
T = **{c['temperature']:.2f}** ({'the detector is over-confident: T above 1 softens it' if c['temperature'] > 1 else 'the detector is under-confident: T below 1 sharpens it'}).

| On the exam set | As the page shows it | After temperature scaling (used by the app) | Platt scaling, a·z + b (comparison only) |
|---|---:|---:|---:|
| Expected calibration error (0 = honest) | {ce['ece']:.3f} | {cc['ece']:.3f} | {cp['ece']:.3f} |
| Brier score (0 = perfect, 0.25 = always guessing 50%) | {ce['brier']:.3f} | {cc['brier']:.3f} | {cp['brier']:.3f} |
| Confidently wrong verdicts (AI↔real) | {ce['confident_wrong_verdicts']} | {cc['confident_wrong_verdicts']} | {cp['confident_wrong_verdicts']} |
| Inconclusive verdicts | {ce['inconclusive_verdicts']} | {cc['inconclusive_verdicts']} | {cp['inconclusive_verdicts']} |

![Reliability diagram](reliability.png)

**What this means.** The detector's percentages are far too sure of themselves: of the images it scores
below 10% AI, {pct(first_bin['fraction_ai'])} really are AI. Once its readings are made honest, it is
almost never sure: {cc['inconclusive_verdicts']} of {cc['n']} exam images land between 20% and 80%. The page's "100%" was
overclaiming; a threshold cannot fix that, better training data can (roadmap Step 4). Until the trust
report (Step 1) re-derives the verdict wording on calibrated numbers, the page keeps its current
verdicts and `/api/analyze` reports the calibrated reading alongside (`calibrated_probability_fake`).

## The source panel ("which tool made it?")

It knows 8 tools. The exam has {a['exam']['known']} AI images from tools it knows
({', '.join(a['known_classes_available'])}) and {a['exam']['unknown']} from tools it does not.

| On the exam set | Before: always names a tool | Now ({naming_rule}) |
|---|---:|---:|
| Known tool, named correctly | {pct(ba['known_named_correctly'])} | {pct(wa['known_named_correctly'])} |
| Known tool, named wrongly | {pct(ba['known_named_wrongly'])} | {pct(wa['known_named_wrongly'])} |
| Known tool, says "can't tell" | {pct(ba['known_said_unknown'])} | {pct(wa['known_said_unknown'])} |
| Unknown tool, says "can't tell" | {pct(ba['unknown_said_unknown'])} | {pct(wa['unknown_said_unknown'])} |
| Unknown tool, confidently names one of its 8 anyway | {pct(ba['unknown_named_a_tool'])} | {pct(wa['unknown_named_a_tool'])} |

How well its top probability separates known from unknown tools (AUROC): calibration set
{a['auroc_top_probability']['calibration']:.3f}, exam set {a['auroc_top_probability']['exam'] or float('nan'):.3f}
(energy score: {a['auroc_energy']['calibration']:.3f} / {a['auroc_energy']['exam'] or float('nan'):.3f}).
**The rule, decided on the calibration set only:** {a['policy']}. (If a retrained model ever does
separate the two, the same rule picks the lowest bar that keeps unknown-tool images named at most
{100 * MAX_UNKNOWN_NAMED:.0f}% of the time.)

**What this means.** {panel_reading(a)}
"""
    if rb:
        md += f"""
## Social-media copies (long side ≤ 1600 px, JPEG quality 75)

{rb['pairs']} exam images were re-saved the way messaging apps do and scored again.
Average change in P(AI): {rb['mean_change_in_p_ai'] * 100:.1f} points (median {rb['median_change_in_p_ai'] * 100:.1f}).
The page's verdict changed for **{rb['verdict_changed']} of {rb['pairs']}**: {', '.join(f'{k}: {v}' for k, v in sorted(rb['verdict_changes'].items())) or 'none'}.
AI images' average P(AI): {pct(rb['ai_images_mean_p_ai']['original']) if rb['ai_images_mean_p_ai'] else '—'} as originals, {pct(rb['ai_images_mean_p_ai']['social_copy']) if rb['ai_images_mean_p_ai'] else '—'} as copies.
AUC: {fmt3(rb['auc']['originals'])} on the originals, {fmt3(rb['auc']['social_copies'])} on the copies.
"""
    rw = s.get("robustness_whatsapp")
    if rw:
        md += f"""
## Real WhatsApp round trips (pictures the team sent and saved back)

{rw['pairs']} pairs. The page's verdict changed for **{rw['verdict_changed']} of {rw['pairs']}**; average change in P(AI):
{rw['mean_change_in_p_ai'] * 100:.1f} points.
"""
    if s.get("user_supplied"):
        md += f"""
## Pictures the team supplied (kept out of the headline, so it stays comparable)

{group_table(s['user_supplied'], 'user')}
"""
    sp = s.get("size_probe")
    if sp:
        md += f"""
## Controlled test: does the size of the file matter?

Before looking, the detector squashes every picture to 256×256 pixels and takes the middle 224×224.
So a picture's size should mostly not reach it — but the small images in this exam are also the ones
from older generators, so it was worth checking. The same {sp['ai_images']} modern AI images and
{sp['real_photos']} large real photos (OpenFake's test split and its DOCCI photos) were shrunk so the long
side is {sp['side_px']} px (nothing else changed) and scored again.

| | Full size | Shrunk to {sp['side_px']} px |
|---|---:|---:|
| AI images called AI-generated | {pct(sp['ai_called_ai']['full_size'])} | {pct(sp['ai_called_ai']['shrunk'])} |
| AI images called authentic | {pct(sp['ai_called_real']['full_size'])} | {pct(sp['ai_called_real']['shrunk'])} |
| Real photos called authentic | {pct(sp['real_called_real']['full_size'])} | {pct(sp['real_called_real']['shrunk'])} |
| Real photos called AI-generated | {pct(sp['real_called_ai']['full_size'])} | {pct(sp['real_called_ai']['shrunk'])} |
| AUC | {sp['auc']['full_size']:.3f} | {sp['auc']['shrunk']:.3f} |

**What this means.** {size_reading(sp)}
"""
    md += """
## Reproduce

```
python evaluation/build_exam_set.py --check     # the images are exactly the ones in the manifest
python evaluation/run_exam.py <label>           # re-score and rewrite this report
```
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    return md


# -------------------------------------------------------------- bias map --

def bias_map_results(s: dict, meta: dict) -> dict:
    def cells(groups):
        out = {}
        for k, v in groups.items():
            out[k] = {"accuracy": v["correct"], "sample_count": v["n"], "trustworthy": v["trustworthy"],
                      "wrong": v["wrong"], "inconclusive": v["inconclusive"], "n_ai": v["n_ai"],
                      "n_real": v["n_real"]}
        return out

    o = s["overall"]
    sections = [
        ("By generator, and by kind of real photo", "AI groups: share called AI-generated. Real groups: share called authentic.", s["by_generator"]),
        ("By where the images came from", "", s["by_source"]),
        ("By kind of generator", "", s["by_kind"]),
        ("By subject", "Subjects are machine-labelled by CLIP, so approximate.", s["by_subject"]),
        ("By image size (longest side)", "", s["by_size"]),
        ("By file format", "", s["by_format"]),
    ]
    rb = s.get("robustness_social_copies")
    if rb:
        sections.append(("Same images, re-saved like a messaging app", f"{rb['pairs']} pairs: long side up to 1600 px, JPEG quality 75.",
                         {"originals": rb["originals"], "social-media copies": rb["social_copies"]}))
    return {
        "generated_at_utc": meta["date_iso"],
        "total_images": o["n"],
        "overall_accuracy": o["correct"],
        "min_trustworthy_group_size": MIN_GROUP,
        "scope_note": (f"Scored on the exam set: {o['n']} labelled images ({o['n_ai']} AI-generated, "
                       f"{o['n_real']} real) from OpenFake's held-out test split, its Reddit set, CNNDetection, "
                       "GenImage, Wikimedia Commons and this project's earlier test images, every label taken from "
                       "its source. A cell's percentage is how often the page's verdict was right; "
                       "'inconclusive' counts as not right. Details: evaluation/BASELINE.md."),
        "headline": {"caught": o["caught"], "false_alarm": o["false_alarm"], "missed": o["missed"],
                     "inconclusive": o["inconclusive"], "auc": o["auc"]},
        "sections": [{"title": t, "note": n, "groups": cells(g)} for t, n, g in sections],
        # Kept so older copies of the page (and the schema test) still find them.
        "by_generator": cells(s["by_generator"]),
        "by_subject": cells(s["by_subject"]),
        "by_style": cells(s["by_kind"]),
        "by_combination": {},
    }


# ------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser(description="Score the exam set and write the report card.")
    parser.add_argument("label")
    parser.add_argument("--fit", action="store_true", help=f"save the fitted temperature and threshold to {CALIBRATION_FILE.name}")
    parser.add_argument("--write-bias-map", action="store_true", help="refresh bias_map/results.json for the Bias Map page")
    parser.add_argument("--reuse", action="store_true", help="re-summarise an existing per_image.jsonl without re-scoring")
    parser.add_argument("--force", action="store_true", help="re-score even though results/<label> already exists")
    args = parser.parse_args()

    out = EVAL / "results" / args.label
    per_image = out / "per_image.jsonl"
    if out.exists() and not (args.reuse or args.force):
        raise SystemExit(f"{out} already exists. Use --reuse to re-summarise its saved scores, "
                         "--force to re-score over it, or pick a new label.")
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse and per_image.exists():
        records = [json.loads(line) for line in per_image.read_text(encoding="utf-8").splitlines()]
        if not any(r["purpose"] == "probe" for r in records):
            records += run_probe_only()   # saved scores predate the size test
    else:
        records = score_all(args.label)
        per_image.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    import subprocess
    git = lambda *a: subprocess.run(["git", *a], cwd=BOUNCER, capture_output=True, text=True).stdout.strip()  # noqa: E731
    now = datetime.now(timezone.utc)
    meta = {"date": now.strftime("%d %b %Y"), "date_iso": now.isoformat(),
            "commit": git("describe", "--always", "--dirty"),  # "-dirty": uncommitted changes were present
            "branch": git("branch", "--show-current")}

    s = summarise(records)
    s["meta"] = meta
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    for r in records:
        if r["set"] == "exam" and r["purpose"] == "detector" and r["label"] in ("ai", "real"):
            r["evidence"] = detector_evidence(r, s["calibration"]["temperature"])
    per_image.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    cols = ["id", "purpose", "label", "source", "generator", "generator_group", "attribution_class", "subject",
            "content", "width", "height", "format", "variant_of", "p_ai_page", "verdict", "logit", "attr_top",
            "attr_top_p", "wobble", "seconds"]
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            if r["set"] == "exam":
                w.writerow({**r, "logit": round(r["logit"], 4),
                            "attr_top_p": round(r["attr_top_p"], 4) if "attr_top_p" in r else ""})

    ai_groups = {k: v for k, v in s["by_generator"].items() if v["n_ai"] and not v["n_real"]}
    real_groups = {k[len("real: "):]: v for k, v in s["by_generator"].items() if v["n_real"] and not v["n_ai"]}
    chart_bars(ai_groups, "caught", "AI images the page calls AI-generated, by generator",
               "% of that generator's exam images", out / "by_generator.png",
               "Exam set. Bars that are short are generators the detector does not recognise.")
    chart_bars(real_groups, "cleared", "Real photos the page calls authentic, by source",
               "% of that source's exam photos", out / "real_sources.png",
               "Exam set. The rest are inconclusive or, worse, called AI-generated.")
    chart_reliability(s["calibration"], out / "reliability.png")
    write_report(args.label, s, out, meta)

    if args.fit:
        CALIBRATION_FILE.write_text(json.dumps({
            "detector_temperature": s["calibration"]["temperature"],
            "attribution_unknown_threshold": s["attribution"]["unknown_threshold"],
            "attribution_score": "top softmax probability; a tool is named only when it is above the threshold",
            "attribution_policy": s["attribution"]["policy"],
            "fitted_on": "evaluation/calibration_set_manifest.csv (never the exam set)",
            "fitted_by": f"evaluation/run_exam.py {args.label} --fit",
            "fitted_at_utc": meta["date_iso"],
            "exam_effect": {"ece_before": s["calibration"]["exam_set"]["raw"]["ece"],
                            "ece_after": s["calibration"]["exam_set"]["calibrated"]["ece"]},
        }, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {CALIBRATION_FILE}")
    if args.write_bias_map:
        BIAS_MAP_RESULTS.write_text(json.dumps(bias_map_results(s, meta), indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {BIAS_MAP_RESULTS}")

    o = s["overall"]
    print(f"\n{o['n']} exam images: caught {pct(o['caught'])}, missed {pct(o['missed'])}, "
          f"false alarms {pct(o['false_alarm'])}, inconclusive {pct(o['inconclusive'])}, AUC {o['auc']:.3f}")
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
