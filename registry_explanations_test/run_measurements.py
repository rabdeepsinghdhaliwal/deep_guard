"""
Deep-Guard -- measurements behind the Content Registry's explanations.

Answers, with numbers, the two questions every threshold in
app/registry_explain.py depends on:

  1. Do REAL copies get confirmed, and is the edit described correctly?
     The same 18 Mona Lisa edits the README already reports, plus a few
     colour edits of the Great Wave (a more colourful painting). For each:
     shared distinctive points, recovered rotation / mirror / area shown /
     added area, and the colour verdict, against the known truth.

  2. How many points do UNRELATED images share by pure chance?
     Every registered-candidate image (the 3 demo paintings + the 8
     labelled test photos) against every other image available locally
     (the same 11 + the 44 MidJourney / DALL-E 3 images in
     generalization_test/): hundreds of pairs that share no content.
     The "confirmed" threshold has to sit well above the largest of these.

Run from the repo root, inside the app's venv (takes about a minute):
    python registry_explanations_test/run_measurements.py
Writes registry_explanations_test/results.json; the README quotes it.
"""

import io
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = REPO / "deepguard-bouncer"
sys.path.insert(0, str(ROOT / "app"))

import local_features as lf  # noqa: E402
import registry_explain as rx  # noqa: E402

ART = ROOT / "demo_artworks"
TEST_PHOTOS = REPO / "test_images"
TEST_PHOTO_FILES = ["ai1.png", "ai2.png", "ai3.png", "ai4.png", "real1.png", "real2.jpg", "real3.jpg", "real4.jpg"]
HELD_OUT = REPO / "generalization_test" / "held_out_images"


def jpeg(im, quality):
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def centre_crop(im, frac):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))


def overlay_text(im):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for i in range(3):
        draw.text((w * 0.1, h * (0.2 + i * 0.3)), "SAMPLE - NOT FOR SALE", fill=(255, 255, 255))
    return im


def sepia(im):
    return ImageOps.colorize(ImageOps.grayscale(im), black=(40, 20, 0), white=(255, 240, 200))


def border(im):
    w, h = im.size
    out = Image.new("RGB", (int(w * 1.3), int(h * 1.3)), (245, 245, 245))
    out.paste(im, (int(w * 0.15), int(h * 0.15)))
    return out


def small_then_big(im):
    w, h = im.size
    return im.resize((max(8, w // 6), max(8, h // 6))).resize((w, h))


# (name, edit, truth). Truth: area = fraction of the original shown,
# rot = degrees anticlockwise, colour = expected verdict.
def edits_for(width, height):
    return [
        ("Same file, re-encoded JPEG q=90", lambda im: jpeg(im, 90), dict(area=1.0, rot=0, mirror=False, colour="kept")),
        ("Heavy compression (JPEG q=10)", lambda im: jpeg(im, 10), dict(area=1.0, rot=0, mirror=False, colour="kept")),
        ("Shrunk to 1/6 size, blown back up", small_then_big, dict(area=1.0, rot=0, mirror=False, colour="kept")),
        ("Crop: centre 80%", lambda im: centre_crop(im, 0.8), dict(area=0.64, rot=0, mirror=False, colour="kept")),
        ("Crop: centre 60%", lambda im: centre_crop(im, 0.6), dict(area=0.36, rot=0, mirror=False, colour="kept")),
        ("Crop: centre 40%", lambda im: centre_crop(im, 0.4), dict(area=0.16, rot=0, mirror=False, colour="kept")),
        ("Crop: centre 25%", lambda im: centre_crop(im, 0.25), dict(area=0.0625, rot=0, mirror=False, colour="kept")),
        ("Crop: top-left quarter", lambda im: im.crop((0, 0, width // 2, height // 2)), dict(area=0.25, rot=0, mirror=False, colour="kept")),
        ("Mirror (left-right flip)", ImageOps.mirror, dict(area=1.0, rot=0, mirror=True, colour="kept")),
        ("Rotate 15 degrees", lambda im: im.rotate(15, expand=True, fillcolor=(255, 255, 255)), dict(area=1.0, rot=15, mirror=False, colour="kept")),
        ("Rotate 90 degrees", lambda im: im.rotate(90, expand=True), dict(area=1.0, rot=90, mirror=False, colour="kept")),
        ("Black & white", lambda im: ImageOps.grayscale(im).convert("RGB"), dict(area=1.0, rot=0, mirror=False, colour="removed")),
        ("Sepia recolour", sepia, dict(area=1.0, rot=0, mirror=False, colour="recoloured")),
        ("Brightness +60%", lambda im: ImageEnhance.Brightness(im).enhance(1.6), dict(area=1.0, rot=0, mirror=False, colour="kept", brightness="brighter")),
        ("Blur (radius 4)", lambda im: im.filter(ImageFilter.GaussianBlur(4)), dict(area=1.0, rot=0, mirror=False, colour="kept")),
        ("Text watermark overlay", overlay_text, dict(area=1.0, rot=0, mirror=False, colour="kept")),
        ("Screenshot-style white border", border, dict(area=1.0, rot=0, mirror=False, colour="kept")),
        ("Combo: crop 60% + B&W + JPEG q=30",
         lambda im: jpeg(ImageOps.grayscale(centre_crop(im, 0.6)).convert("RGB"), 30),
         dict(area=0.36, rot=0, mirror=False, colour="removed")),
    ]


def features(image, max_points):
    frame = lf.examination_frame(image)
    feats = lf.extract(frame, image.size, max_points)
    return frame, feats


def examine_copy(reference_feats, edited):
    t0 = time.perf_counter()
    frame, query = features(edited, lf.MAX_QUERY_POINTS)
    verification = lf.verify(query, reference_feats, lf.mirror(query, frame, lf.MAX_QUERY_POINTS))
    ms = (time.perf_counter() - t0) * 1000
    row = {"shared_points": verification.shared_points, "ms": round(ms)}
    if verification.shared_points >= rx.CONFIRMED_POINTS:
        geometry = lf.describe_geometry(verification, query, reference_feats)
        colour = lf.compare_colour(frame, verification, reference_feats)
        row.update({k: geometry[k] for k in ("rotation_deg", "mirrored", "size_ratio", "original_area_shown",
                                             "shown_position", "added_area")})
        row.update(colour=colour["change"], brightness=colour["brightness"])
    return row


def copies_section():
    rows = []
    for painting, only in [("mona_lisa", None), ("great_wave", {"Same file, re-encoded JPEG q=90", "Heavy compression (JPEG q=10)",
                                                              "Black & white", "Sepia recolour", "Brightness +60%",
                                                              "Combo: crop 60% + B&W + JPEG q=30"})]:
        original = Image.open(ART / f"{painting}.jpg").convert("RGB")
        _, reference = features(original, lf.MAX_STORED_POINTS)
        for name, edit, truth in edits_for(*original.size):
            if only is not None and name not in only:
                continue
            row = {"painting": painting, "edit": name, "truth": truth, **examine_copy(reference, edit(original))}
            row["confirmed"] = row["shared_points"] >= rx.CONFIRMED_POINTS
            if row["confirmed"]:
                row["correct"] = {
                    "area": abs(row["original_area_shown"] - truth["area"]) <= 0.03,
                    "rotation": abs(row["rotation_deg"] - truth["rot"]) <= 2,
                    "mirror": row["mirrored"] == truth["mirror"],
                    "colour": row["colour"] == truth["colour"],
                    "brightness": row["brightness"] == truth.get("brightness", "same"),
                }
            rows.append(row)
    return rows


def unrelated_section():
    candidates = {p.name: p for p in sorted(ART.glob("*.jpg"))}
    candidates.update({name: TEST_PHOTOS / name for name in TEST_PHOTO_FILES})
    queries = dict(candidates)
    for folder in sorted(HELD_OUT.iterdir()):
        for p in sorted(folder.iterdir()):
            queries[f"{folder.name}/{p.name}"] = p

    references = {}
    for name, path in candidates.items():
        references[name] = features(Image.open(path).convert("RGB"), lf.MAX_STORED_POINTS)[1]
    pairs = []
    for qname, qpath in queries.items():
        frame, query = features(Image.open(qpath).convert("RGB"), lf.MAX_QUERY_POINTS)
        mirrored = lf.mirror(query, frame, lf.MAX_QUERY_POINTS)
        for rname, reference in references.items():
            if rname == qname:
                continue
            pairs.append({"reference": rname, "query": qname,
                          "shared_points": lf.verify(query, reference, mirrored).shared_points})
    return pairs


def main():
    t0 = time.perf_counter()
    copies = copies_section()
    unrelated = unrelated_section()
    shared = sorted(p["shared_points"] for p in unrelated)
    real = [r["shared_points"] for r in copies]
    summary = {
        "copies_tested": len(copies),
        "copies_confirmed": sum(r["confirmed"] for r in copies),
        "copies_fewest_shared_points": min(real),
        "copies_all_facts_correct": sum(1 for r in copies if r.get("correct") and all(r["correct"].values())),
        "unrelated_pairs": len(unrelated),
        "unrelated_max_shared_points": shared[-1],
        "unrelated_mean_shared_points": round(statistics.mean(shared), 2),
        "unrelated_pairs_at_or_above_weak": sum(s >= rx.WEAK_POINTS for s in shared),
        "unrelated_pairs_at_or_above_confirmed": sum(s >= rx.CONFIRMED_POINTS for s in shared),
        "thresholds": {"weak": rx.WEAK_POINTS, "confirmed": rx.CONFIRMED_POINTS, "strong": rx.STRONG_POINTS},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "copies": copies, "unrelated": unrelated}, indent=1),
                                       encoding="utf-8")

    print(f"{'painting':10s} {'edit':36s} {'pts':>5s} {'rot':>6s} {'mir':>4s} {'shown':>6s} {'added':>6s} {'colour':>11s} {'bright':>9s} ok")
    for r in copies:
        ok = "yes" if r.get("correct") and all(r["correct"].values()) else ("NOT CONFIRMED" if not r["confirmed"] else
                                                                        [k for k, v in r["correct"].items() if not v])
        print(f"{r['painting'][:10]:10s} {r['edit'][:36]:36s} {r['shared_points']:5d} {r.get('rotation_deg', 0):6.1f} "
              f"{str(r.get('mirrored', ''))[:1]:>4s} {r.get('original_area_shown', 0):6.2f} {r.get('added_area', 0):6.2f} "
              f"{r.get('colour', ''):>11s} {r.get('brightness', ''):>9s} {ok}")
    top = sorted(unrelated, key=lambda p: -p["shared_points"])[:5]
    print(f"\nUnrelated pairs: {len(unrelated)}; most shared points {shared[-1]}; mean {summary['unrelated_mean_shared_points']}")
    for p in top:
        print(f"   {p['shared_points']:3d}  {p['reference']:16s} <- {p['query']}")
    print("\nsummary:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
