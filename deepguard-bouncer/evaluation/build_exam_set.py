"""
Deep-Guard -- builds the exam set and the calibration set.

The exam set is the fixed, labelled collection that every roadmap step is
measured on -- "the exam paper". The calibration set is a separate,
smaller collection used only to *fit* things (the detector's temperature,
the source panel's "unknown" threshold). Fitting on the exam set would be
setting the exam from the answer key: the scores would look better than
the system is.

Rules this script enforces:
  1. Labels are never guessed. Every image comes from a dataset that
     records whether it is real or AI (and which generator made it), from
     this project's own earlier test sets, or from a Wikimedia Commons page
     that states it.
  2. The two sets never share an image (checked by SHA-256). Exam images
     come from OpenFake's *test* splits; Step 4 will train on its *train*
     split, so nothing here can leak into training.
  3. Rebuildable. Images live outside the repository, in $DEEPGUARD_DATA
     (default ~/deepguard-data) -- a few GB that do not belong in git or
     in a synced OneDrive folder. The manifests next to this file record
     where every image came from, pinned to a dataset revision, with its
     SHA-256, so a rebuild can be checked byte for byte.

Usage (from deepguard-bouncer/, after pip install -r evaluation/requirements-eval.txt):
    python evaluation/build_exam_set.py           # build, or top up
    python evaluation/build_exam_set.py --check   # every file present and unchanged?

Pictures only you can supply (your phone photos, WhatsApp round trips,
ChatGPT / Gemini images) go in $DEEPGUARD_DATA/exam_set/incoming/<folder>/.
The folders are created on the first run, each with a note on what belongs
there; the next run adds whatever is in them to the manifest.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from PIL import Image, ImageOps

BOUNCER = Path(__file__).resolve().parents[1]
REPO = BOUNCER.parent
EVAL = BOUNCER / "evaluation"
DATA = Path(os.environ.get("DEEPGUARD_DATA", Path.home() / "deepguard-data"))
EXAM_DIR = DATA / "exam_set"
CAL_DIR = DATA / "calibration_set"
INDEX_DIR = DATA / "openfake_index"
EXAM_MANIFEST = EVAL / "exam_set_manifest.csv"
CAL_MANIFEST = EVAL / "calibration_set_manifest.csv"

SEED = 20260927
# Wikimedia refuses requests whose User-Agent carries no contact address.
UA = {"User-Agent": "DeepGuardEvaluation/1.0 (student research project; "
                    "https://github.com/rabdeepsinghdhaliwal/deep_guard)"}

# Every remote source is pinned to one revision, so "rebuild" means "the
# same bytes", not "whatever the dataset says today".
OPENFAKE = "datasets/ComplexDataLab/OpenFake@99afb8811d6384b0a014a6dad7ae85a579e29b32"
CNNDET = "datasets/sywang/CNNDetection@f287cf3c3e92ca2e5451ce200bab8b6f5e0175b2"
GENIMAGE = "datasets/jzousz/GenImage@71c983e6262684bc2c6b6af99582e8f568c259a5"
C2PA_BASE = ("https://raw.githubusercontent.com/c2pa-org/public-testfiles/"
             "22beccc075707475b038d8789d0136c009e43143/legacy/1.4/")
LICENCES = {
    "openfake": "CC-BY-NC-4.0 (OpenFake, Livernoche et al. 2025)",
    "cnndetection": "CC-BY-NC-SA-4.0 (CNNDetection, Wang et al. 2020)",
    "genimage": "non-commercial research use (GenImage, Zhu et al. 2023)",
    "c2pa": "CC-BY-SA-4.0 (C2PA public test files)",
}

FIELDS = ["id", "set", "purpose", "file", "label", "source", "source_ref", "generator",
          "generator_group", "attribution_class", "content", "subject", "width", "height",
          "format", "bytes", "sha256", "licence", "credit", "variant_of", "transform",
          "expected", "notes", "overlap_risk"]

# ------------------------------------------------------------ labelling help --

# The 8 generators the source panel was trained on (app/model.py). A row
# gets attribution_class only when the generator really is one of them.
ATTRIBUTION_MAP = {
    ("cnndetection", "stylegan2"): "stylegan2",
    ("cnndetection", "progan"): "pro_gan",
    ("cnndetection", "biggan"): "big_gan",
    ("cnndetection", "cyclegan"): "cycle_gan",
    ("genimage", "glide_imagenet"): "glide",
    ("genimage", "sdv4_imagenet"): "stable_diffusion",
    ("genimage", "sdv5_imagenet"): "stable_diffusion",
    ("genimage", "biggan_imagenet"): "big_gan",
}
OPENFAKE_SD1 = {"sd-1.4", "sd-1.5", "sd-2.1", "stable-diffusion-v1-4", "stable-diffusion-v1-5"}

VIDEO_MODELS = {"veo-3", "sora-2", "wan-video-2.5", "wan-video-2.2", "kling", "hunyuan-video"}
CLOSED_MODELS = {"gpt-image-1", "gpt-image-1.5", "gpt-image-2", "nano-banana", "nano-banana-pro",
                 "midjourney-6", "midjourney-7", "midjourney-5", "seedream-v5.0", "seedream-v4.0",
                 "seedream-v4.5", "recraft-v2", "recraft-v3", "ideogram-2.0", "ideogram-3.0",
                 "imagen-3.0-002", "imagen-4", "dalle-3", "flux-1.1-pro", "grok-2-image-1212"}


def generator_group(source: str, generator: str, label: str) -> str:
    if label == "real":
        return "real photo"
    if source in ("cnndetection", "genimage") and generator != "midjourney_imagenet":
        return "older research model"
    if generator in VIDEO_MODELS:
        return "video-model frame"
    if generator in CLOSED_MODELS or generator == "midjourney_imagenet":
        return "closed commercial"
    if generator in ("flux.2-klein-9b", "z-image-turbo", "illustrious") or generator in OPENFAKE_SD1 \
            or generator.startswith("stable-diffusion"):
        return "open weights"
    return "other / unknown"


# --------------------------------------------------------------- utilities --

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_facts(data: bytes) -> tuple[int, int, str]:
    im = Image.open(io.BytesIO(data))
    return im.width, im.height, (im.format or "").upper()


EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "MPO": ".jpg", "BMP": ".bmp"}


def safe_stem(stem: str) -> str:
    """Source names can hold ':' and spaces; Windows filenames can't."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def save(set_dir: Path, sub: str, stem: str, data: bytes) -> tuple[str, dict]:
    """Writes the original bytes untouched (no re-encoding: re-encoding would
    change exactly the traces a detector looks at), named by true format."""
    w, h, fmt = image_facts(data)
    stem = safe_stem(stem)
    rel = f"{sub}/{stem}{EXT.get(fmt, '.bin')}"
    path = set_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    elif sha256(path.read_bytes()) != sha256(data):
        # Two different images mapped to one file name: refuse, never let a
        # manifest row point at someone else's picture.
        raise SystemExit(f"name collision: {path} already holds a different image")
    return rel, {"width": w, "height": h, "format": fmt, "bytes": len(data), "sha256": sha256(data)}


def row(**kw) -> dict:
    r = {f: "" for f in FIELDS}
    r.update({k: v for k, v in kw.items() if k in r})
    return r


def hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


# ---------------------------------------------------------------- OpenFake --

def openfake_index(shard: str) -> list[dict]:
    """label/model/type for every row of one Parquet shard, cached locally.
    Reading only these small columns costs ~1 minute per shard instead of
    downloading its ~5 GB of pictures."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = INDEX_DIR / (shard.replace("/", "__").replace(".parquet", ".csv"))
    if not cache.exists():
        import pyarrow.parquet as pq
        print(f"  indexing {shard} ...", flush=True)
        f = pq.ParquetFile(hf_fs().open(f"{OPENFAKE}/{shard}", block_size=2**20))
        rows = []
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g, columns=["label", "model", "type", "release_date"])
            cols = [t.column(c).to_pylist() for c in ("label", "model", "type", "release_date")]
            rows += [[shard, g, i, *vals] for i, vals in enumerate(zip(*cols))]
        with open(cache, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["shard", "row_group", "row", "label", "model", "type", "release_date"])
            w.writerows(rows)
    with open(cache, newline="", encoding="utf-8") as fh:
        return [dict(r, row_group=int(r["row_group"]), row=int(r["row"])) for r in csv.DictReader(fh)]


def pick_row_groups(index: list[dict], key, quotas: dict, max_groups: int) -> list[tuple]:
    """Greedy: repeatedly take the row group that fills the most unmet quota,
    counting rare keys more, so a handful of ~100 MB downloads covers every
    generator instead of fetching whole 5 GB shards."""
    per_group = defaultdict(Counter)
    for r in index:
        per_group[(r["shard"], r["row_group"])][key(r)] += 1
    total = Counter(key(r) for r in index)
    weight = {k: 1.0 / max(total[k], 1) for k in quotas}
    remaining, chosen = dict(quotas), []
    while len(chosen) < max_groups and any(v > 0 for v in remaining.values()):
        def gain(g):
            return sum(min(remaining.get(k, 0), c) * weight.get(k, 0) for k, c in per_group[g].items())
        best = max((g for g in per_group if g not in chosen), key=gain, default=None)
        if best is None or gain(best) <= 0:
            break
        chosen.append(best)
        for k, c in per_group[best].items():
            if k in remaining:
                remaining[k] = max(0, remaining[k] - c)
    return chosen


def openfake_part(set_name: str, set_dir: Path, sub: str, shards: list[str], key, quotas: dict,
                  max_groups: int, rng: random.Random, source: str, taken: set) -> list[dict]:
    import pyarrow.parquet as pq
    index = [r for s in shards for r in openfake_index(s)]
    groups = pick_row_groups(index, key, quotas, max_groups)
    candidates = defaultdict(list)
    for r in index:
        if (r["shard"], r["row_group"]) in groups and key(r) in quotas:
            candidates[key(r)].append(r)
    wanted = set()
    for k, quota in quotas.items():
        pool = sorted(candidates[k], key=lambda r: (r["shard"], r["row_group"], r["row"]))
        rng.shuffle(pool)
        wanted |= {(r["shard"], r["row_group"], r["row"]) for r in pool[:quota]}
    print(f"  {set_name}/{sub}: {len(wanted)} images from {len(groups)} row groups", flush=True)

    rows, fs = [], hf_fs()
    by_shard = defaultdict(list)
    for shard, g in groups:
        by_shard[shard].append(g)
    meta_of = {(r["shard"], r["row_group"], r["row"]): r for r in index}
    for shard, gs in by_shard.items():
        f = pq.ParquetFile(fs.open(f"{OPENFAKE}/{shard}", block_size=8 * 2**20))
        for g in sorted(gs):
            needed = sorted(i for s, gg, i in wanted if s == shard and gg == g)
            stem0 = f"{shard.split('/')[0]}-{shard.split('-')[1]}-rg{g:02d}"
            on_disk = {i: next((set_dir / sub).glob(f"{stem0}-r{i:03d}.*"), None) for i in needed}
            t = None
            if any(p is None for p in on_disk.values()):  # else: all there from an earlier run
                t0 = time.time()
                t = f.read_row_group(g, columns=["image"])
                print(f"    {shard} rg{g}: {t.nbytes / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
            for i in needed:
                data = on_disk[i].read_bytes() if on_disk[i] else t.column("image")[i].as_py()["bytes"]
                meta = meta_of[(shard, g, i)]
                h = sha256(data)
                if h in taken:
                    continue
                taken.add(h)
                rel, facts = save(set_dir, sub, f"{stem0}-r{i:03d}", data)
                label = "real" if meta["label"] == "real" else "ai"
                if label == "real":
                    gen, credit = "", ("real photo from a photography subreddit" if meta["model"] == "reddit"
                                       else f"real photo ({meta['model']})")
                elif meta["model"] == "reddit":
                    gen, credit = "unknown", "posted in an AI-generation subreddit (tool not recorded)"
                else:
                    gen, credit = meta["model"], f"generated by {meta['model']}"
                rows.append(row(
                    id=f"{sub}-{shard.split('-')[1]}-{g:02d}-{i:03d}", set=set_name, purpose="detector",
                    file=rel, label=label, source=source,
                    source_ref=f"{OPENFAKE}/{shard}#rowgroup={g}&row={i}",
                    generator=gen, generator_group=generator_group(source, gen, label),
                    attribution_class="stable_diffusion" if gen in OPENFAKE_SD1 else "",
                    content="video frame" if meta["type"] == "video" else "image",
                    licence=LICENCES["openfake"], credit=credit,
                    notes=f"openfake type={meta['type']}; release={meta['release_date']}", **facts))
    return rows


# --------------------------------------------------- CNNDetection / GenImage --

def zip_part(set_name: str, set_dir: Path, sub: str, repo_zip: str, picks: dict, source: str,
             rng: random.Random, taken: set, exclude: set) -> list[dict]:
    """picks: {(folder_prefix, label): count}. Reads only the zip's table of
    contents and the chosen members, never the 20-25 GB archive."""
    z = zipfile.ZipFile(hf_fs().open(repo_zip, block_size=2**20))
    members = [m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith((".png", ".jpg", ".jpeg"))]
    rows = []
    for (prefix, label), count in picks.items():
        # CNNDetection keeps each generator's outputs in 1_fake/ and the real
        # images it was compared against in 0_real/ (sometimes one level down,
        # per LSUN category); GenImage's prefixes already end in ai/ or nature/.
        marker = {"ai": "/1_fake/", "real": "/0_real/"}[label] if source == "cnndetection" else ""
        pool = sorted(m.filename for m in members if m.filename.startswith(prefix)
                      and marker in m.filename and m.filename not in exclude)
        rng.shuffle(pool)
        got = 0
        for name in pool:
            if got >= count:
                break
            folder = prefix.strip("/").split("/")[-2] if source == "genimage" else prefix.split("/")[0]
            # The whole member path: CNNDetection repeats file names across its
            # per-category folders (progan/airplane/1_fake/11975.png and
            # progan/bicycle/1_fake/11975.png are different images).
            stem = Path(name).with_suffix("").as_posix().replace("/", "-")[-120:]
            local = next((set_dir / sub).glob(f"{safe_stem(stem)}.*"), None)  # from an earlier run
            data = local.read_bytes() if local else z.read(name)
            h = sha256(data)
            if h in taken:
                continue
            taken.add(h)
            exclude.add(name)
            rel, facts = save(set_dir, sub, stem, data)
            gen = "" if label == "real" else folder
            attribution_class = ATTRIBUTION_MAP.get((source, folder), "") if label == "ai" else ""
            # Known ways an exam image might not be new to our models. Flagged,
            # never hidden: run_exam.py keeps "detector" risks out of the
            # headline and calls the source panel's known-tool score an upper bound.
            risks = []
            if folder == "whichfaceisreal":
                risks.append("detector: FFHQ real faces and StyleGAN faces are in its training data "
                             "(Kaggle 140k-real-and-fake-faces)")
            if attribution_class:
                risks.append("source panel: its training set (ArtiFact) collected public GAN/diffusion "
                             "outputs and may include this image")
            rows.append(row(
                id=f"{sub}-{folder}-{label}-{got:02d}", set=set_name, purpose="detector", file=rel,
                label=label, source=source, source_ref=f"{repo_zip}#{name}", generator=gen,
                generator_group=generator_group(source, gen, label), attribution_class=attribution_class,
                content="image", licence=LICENCES[source],
                credit=f"{source} test set, {folder} {'real counterpart' if label == 'real' else 'output'}",
                overlap_risk="; ".join(risks), **facts))
            got += 1
        print(f"  {set_name}/{sub}: {got} x {prefix} ({label})", flush=True)
    return rows


# ------------------------------------------------------------ local sources --

COMMONS_GENERATOR = {
    "ai_dalle3_hungry_man": "dalle-3", "ai_dalle3_orange_cat": "dalle-3",
    "ai_dalle3_astronaut_mercury": "dalle-3", "ai_dalle3_cat_wikipedia": "dalle-3",
    "ai_sd35_cyberpunk_chef": "stable-diffusion-3.5", "ai_sd_man_idea": "stable-diffusion (version not stated)",
    "ai_photo_programming_class": "unknown", "ai_stylegan_man": "stylegan (version not stated)",
    "ai_stylegan_woman": "stylegan (version not stated)",
}
# The Commons pictures: which files, with credits and licences, are listed in
# the repo (commons_sources.json, with each original's width); the pictures
# themselves are fetched from Commons on first build. For originals wider
# than 2000 px Commons serves its own resized copy (it picks the size), as
# when they were first downloaded for the viva demo.
COMMONS_SOURCES = EVAL / "commons_sources.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def fetch_commons(title: str) -> bytes:
    client = httpx.Client(headers=UA, timeout=120, follow_redirects=True)
    info = client.get(COMMONS_API, params={"action": "query", "format": "json", "prop": "imageinfo",
                                           "iiprop": "url|size", "iiurlwidth": 2000, "titles": title}).json()
    ii = next(iter(info["query"]["pages"].values()))["imageinfo"][0]
    r = client.get(ii["thumburl"] if ii.get("width", 0) > 2000 else ii["url"])
    r.raise_for_status()
    return r.content


def local_part(taken: set) -> list[dict]:
    rows = []
    # 1. The 44 images from generalization_test/ (official DALL-E 3 examples
    #    and the MidJourney showcase) -- this project's own earlier test set.
    for gen_dir, gen, credit in (("dalle3", "dalle-3", "OpenAI's official DALL-E 3 examples page"),
                                 ("midjourney", "midjourney-5", "midjourney.com showcase (v5)")):
        for p in sorted((REPO / "generalization_test" / "held_out_images" / gen_dir).iterdir()):
            data = p.read_bytes()
            taken.add(sha256(data))
            rel, facts = save(EXAM_DIR, "outside", p.stem, data)
            rows.append(row(id=f"outside-{p.stem}", set="exam", purpose="detector", file=rel, label="ai",
                            source="deepguard-generalization-test", source_ref=f"generalization_test/held_out_images/{gen_dir}/{p.name}",
                            generator=gen, generator_group="closed commercial", content="image",
                            licence="copyright of the generator's owner; used for evaluation only",
                            credit=credit, **facts))
    # 2. The project's original 8 test images (bias map v1). a12.png is left out on purpose.
    with open(BOUNCER / "bias_map" / "labeled_test_set.csv", encoding="utf-8") as fh:
        labels = list(csv.DictReader(fh))
    for r in labels:
        p = REPO / "test_images" / r["filename"]
        data = p.read_bytes()
        taken.add(sha256(data))
        rel, facts = save(EXAM_DIR, "project", p.stem, data)
        label = "real" if r["true_label"] == "real" else "ai"
        rows.append(row(id=f"project-{p.stem}", set="exam", purpose="detector", file=rel, label=label,
                        source="deepguard-original-test-images", source_ref=f"test_images/{p.name}",
                        generator="" if label == "real" else "unknown",
                        generator_group="real photo" if label == "real" else "other / unknown",
                        content="image", licence="unknown (project's original test images)",
                        credit="project's original 8-image test set",
                        overlap_risk="detector: the project's demo images, looked at throughout development",
                        **facts))
    # 3. Wikimedia Commons images with stated licences (India-relevant real
    #    photos, and AI images whose Commons page names the generator).
    commons_dir = EXAM_DIR / "commons"
    credits = json.loads(COMMONS_SOURCES.read_text(encoding="utf-8"))
    for key, c in sorted(credits.items()):
        path = next((p for p in commons_dir.glob(f"{key}.*") if p.suffix != ".json"), None)
        if path is None:  # first build on this machine: fetch it from Commons
            save(EXAM_DIR, "commons", key, fetch_commons(c["commons_title"]))
            path = next(p for p in commons_dir.glob(f"{key}.*") if p.suffix != ".json")
        data = path.read_bytes()
        taken.add(sha256(data))
        rel, facts = save(EXAM_DIR, "commons", key, data)
        label = "ai" if key.startswith("ai_") else "real"
        gen = COMMONS_GENERATOR.get(key, "") if label == "ai" else ""
        original = c.get("original_width")
        resized = bool(original) and facts["width"] < original
        rows.append(row(id=f"commons-{key}", set="exam", purpose="detector", file=rel, label=label,
                        source="wikimedia-commons", source_ref=c["source"], generator=gen,
                        generator_group=("older research model" if "stylegan" in gen else
                                         generator_group("commons", gen, label)),
                        content="image", licence=c["licence"], credit=c["author"],
                        notes=(f"Commons' own resized copy ({facts['width']} px wide) of a {original} px original"
                               if resized else ""),
                        **facts))
    return rows


# --------------------------------------------------------- C2PA test files --

C2PA_EXPECTED = {
    # no manifest at all
    "adobe-20220124-A.jpg": "none", "adobe-20220124-I.jpg": "none",
    # deliberately broken manifests (the file's own name says how)
    "adobe-20220124-CIE-sig-CA.jpg": "invalid: an ingredient's signature does not validate",
    "adobe-20220124-E-clm-CAICAI.jpg": "invalid: a referenced claim is missing",
    "adobe-20220124-E-dat-CA.jpg": "invalid: hard-binding hash mismatch (pixels changed after signing)",
    "adobe-20220124-E-sig-CA.jpg": "invalid: signature does not validate",
    "adobe-20220124-E-uri-CA.jpg": "invalid: an assertion was tampered with",
    "adobe-20220124-E-uri-CIE-sig-CA.jpg": "invalid: tampered assertion and bad ingredient signature",
    "adobe-20220124-XCA.jpg": "invalid: hash mismatch",
    "adobe-20220124-XCI.jpg": "invalid: hash mismatch",
    "nikon-20221019-building.jpeg": "invalid: claim signature mismatch",
    # camera captures signed by Truepic
    "truepic-20230212-camera.jpg": "valid: camera capture",
    "truepic-20230212-landscape.jpg": "valid: camera capture",
    "truepic-20230212-library.jpg": "valid: camera capture",
}
C2PA_VALID_ADOBE = ["C", "CA", "CACA", "CACAICAICICA", "CAI", "CAIAIIICAICIICAIICICA", "CAICA",
                    "CAICAI", "CI", "CICA", "CICACACA", "CII"]


def c2pa_part(taken: set) -> list[dict]:
    names = dict(C2PA_EXPECTED)
    names.update({f"adobe-20220124-{c}.jpg": "valid: signed with the C2PA test certificate (not on any trust list)"
                  for c in C2PA_VALID_ADOBE})
    rows, client = [], httpx.Client(headers=UA, timeout=120, follow_redirects=True)
    for name, expected in sorted(names.items()):
        dest = EXAM_DIR / "c2pa" / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            r = client.get(C2PA_BASE + "image/jpeg/" + name)
            r.raise_for_status()
            dest.write_bytes(r.content)
        data = dest.read_bytes()
        taken.add(sha256(data))
        w, h, fmt = image_facts(data)
        rows.append(row(id=f"c2pa-{Path(name).stem}", set="exam", purpose="layer0", file=f"c2pa/{name}",
                        label="real" if name.startswith(("truepic", "nikon")) else "",
                        source="c2pa-public-testfiles", source_ref=C2PA_BASE + "image/jpeg/" + name,
                        content="image", licence=LICENCES["c2pa"], credit="C2PA public test files",
                        expected=json.dumps({"c2pa": expected}), width=w, height=h, format=fmt,
                        bytes=len(data), sha256=sha256(data),
                        notes="" if name.startswith(("truepic", "nikon")) else
                        "label left blank: an Adobe test composite, not a clean real/AI case"))
    video = "truepic-20230212-zoetrope.mp4"
    dest = EXAM_DIR / "c2pa" / video
    if not dest.exists():
        r = client.get(C2PA_BASE + "video/mp4/" + video)
        r.raise_for_status()
        dest.write_bytes(r.content)
    data = dest.read_bytes()
    rows.append(row(id="c2pa-truepic-20230212-zoetrope", set="exam", purpose="layer0", file=f"c2pa/{video}",
                    label="real", source="c2pa-public-testfiles", source_ref=C2PA_BASE + "video/mp4/" + video,
                    content="video", licence=LICENCES["c2pa"], credit="C2PA public test files",
                    expected=json.dumps({"c2pa": "valid: camera capture (video)"}), format="MP4",
                    bytes=len(data), sha256=sha256(data)))
    return rows


# ------------------------------------------------------ derived variants --

def social_copy(data: bytes) -> bytes:
    """What a messaging app typically does to a photo: long side at most
    1600 px, JPEG quality 75, metadata dropped. A *simulation*: real
    WhatsApp round trips go in incoming/ and are labelled as such."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(im.size) > 1600:
        im.thumbnail((1600, 1600), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=75)
    return out.getvalue()


def social_part(originals: list[dict], rng: random.Random, taken: set, per_label: int = 40) -> list[dict]:
    rows = []
    for label in ("real", "ai"):
        pool = sorted((r for r in originals if r["label"] == label and r["purpose"] == "detector"),
                      key=lambda r: r["id"])
        rng.shuffle(pool)
        for r in pool[:per_label]:
            data = social_copy((EXAM_DIR / r["file"]).read_bytes())
            h = sha256(data)
            if h in taken:
                continue
            taken.add(h)
            rel, facts = save(EXAM_DIR, "social_sim", f"{r['id']}-social", data)
            rows.append(row(**{**r, **facts, "id": f"{r['id']}-social", "file": rel, "purpose": "robustness",
                               "variant_of": r["id"], "transform": "social-sim: long side <=1600, JPEG q75",
                               "notes": "derived copy of variant_of; scored separately, never counted "
                                        "as an independent image"}))
    return rows


# ------------------------------------------------------- Layer 1 positives --

VIDEOSEAL_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit"
# The 48-bit message Stable Diffusion XL's reference pipeline stamps on every
# image (diffusers' StableDiffusionXLWatermarker) -- public by design.
SDXL_BITS = [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]]


def watermark_part(exam_rows: list[dict], rng: random.Random, taken: set) -> list[dict]:
    """Known positives for Layer 1: 12 exam pictures (6 real, 6 AI) each
    carrying one of three open watermarks, plus a social-media-like copy of
    every one (does the mark survive?). Everything else in the exam set --
    including these 12 pictures unmarked -- is a known negative.

      trustmark    Adobe TrustMark (variant Q, 61-bit payload), the mark used
                   for "durable" Content Credentials
      videoseal    Meta Video Seal (256-bit payload), TorchScript model
      sdxl-dwtdct  Stable Diffusion XL's reference mark (invisible-watermark,
                   dwtDct, SDXL's public 48-bit message). Note: this method
                   reads back only ~81-85% of its bits even from an untouched
                   copy (measured 27 Sep 2026) -- a real, known weakness.
    """
    try:
        import cv2
        import numpy as np
        import torch
        from imwatermark import WatermarkEncoder
        from trustmark import TrustMark
    except ImportError as exc:
        print(f"  [skipped] watermark tools not installed ({exc}); see requirements-eval.txt")
        return []
    jit = DATA / "models" / "videoseal_y_256b_img.jit"
    if not jit.exists():
        jit.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", VIDEOSEAL_URL, headers=UA, timeout=600, follow_redirects=True) as r:
            r.raise_for_status()
            with open(jit, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)

    bases = []
    for label in ("real", "ai"):
        pool = sorted((r for r in exam_rows if r["source"] == "openfake-core-test" and r["label"] == label),
                      key=lambda r: r["id"])
        rng.shuffle(pool)
        bases += pool[:6]

    tm = TrustMark(verbose=False, model_type="Q")
    vs = torch.jit.load(str(jit)).eval()
    rows = []
    for base in bases:
        im = Image.open(EXAM_DIR / base["file"]).convert("RGB")
        im.thumbnail((1600, 1600), Image.LANCZOS)  # keeps CPU embedding time sane
        tm_bits = "".join(rng.choice("01") for _ in range(61))
        vs_bits = [rng.randint(0, 1) for _ in range(256)]
        marked = {}
        stem = f"{base['id']}"
        if not (EXAM_DIR / "watermark" / f"{stem}-trustmark.png").exists():
            marked["trustmark"] = (tm.encode(im, tm_bits, MODE="binary").convert("RGB"), tm_bits)
        else:
            marked["trustmark"] = (None, tm_bits)
        if not (EXAM_DIR / "watermark" / f"{stem}-videoseal.png").exists():
            x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().unsqueeze(0) / 255
            with torch.no_grad():
                xw = vs.embed(x, torch.tensor([vs_bits], dtype=torch.float32)).clamp(0, 1)
            arr = (xw[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            marked["videoseal"] = (Image.fromarray(arr), "".join(map(str, vs_bits)))
        else:
            marked["videoseal"] = (None, "".join(map(str, vs_bits)))
        if not (EXAM_DIR / "watermark" / f"{stem}-sdxl-dwtdct.png").exists():
            enc = WatermarkEncoder()
            enc.set_watermark("bits", SDXL_BITS)
            bgr = enc.encode(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), "dwtDct")
            marked["sdxl-dwtdct"] = (Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), "".join(map(str, SDXL_BITS)))
        else:
            marked["sdxl-dwtdct"] = (None, "".join(map(str, SDXL_BITS)))

        for method, (pic, message) in marked.items():
            path = EXAM_DIR / "watermark" / f"{stem}-{method}.png"
            if pic is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                pic.save(path, "PNG")
            data = path.read_bytes()
            for variant, blob, transform in (("", data, f"watermark:{method}"),
                                             ("-social", social_copy(data),
                                              f"watermark:{method} + social-sim: long side <=1600, JPEG q75")):
                h = sha256(blob)
                if h in taken:
                    continue
                taken.add(h)
                rel, facts = save(EXAM_DIR, "watermark", f"{stem}-{method}{variant}", blob)
                rows.append(row(**{**base, **facts, "id": f"wm-{method}{variant}-{stem}", "file": rel,
                                   "purpose": "layer1", "variant_of": base["id"], "transform": transform,
                                   "expected": json.dumps({"watermark": method, "message_bits": message}),
                                   "notes": "known watermark positive, made by this script from variant_of"}))
    print(f"  {len(rows)} watermarked files from {len(bases)} base pictures")
    return rows


INCOMING = {
    "phone_real": ("real", "Photos straight from your own phones (copied by USB or Google Drive, not WhatsApp). "
                           "JPEG or PNG only: on an iPhone set Settings > Camera > Formats > Most Compatible "
                           "first (HEIC photos are skipped)."),
    "chatgpt_ai": ("ai", "Images made in ChatGPT, downloaded straight from the app (not via WhatsApp)."),
    "gemini_ai": ("ai", "Images made in Gemini, downloaded straight from the app (not via WhatsApp)."),
    "whatsapp_real": ("real", "Real photos after a WhatsApp round trip: send a photo from phone_real to yourself "
                              "on WhatsApp, save it back, and give it THE SAME FILE NAME as the original in "
                              "phone_real -- that is how the two are paired."),
    "whatsapp_ai": ("ai", "AI images after a WhatsApp round trip: send an image from chatgpt_ai or gemini_ai "
                          "to yourself, save it back, and give it THE SAME FILE NAME as the original."),
}
# A WhatsApp copy is paired with the original of the same file name in these folders.
WHATSAPP_ORIGINALS = {"whatsapp_real": ("phone_real",), "whatsapp_ai": ("chatgpt_ai", "gemini_ai")}
USER_GENERATOR = {"chatgpt_ai": "chatgpt (OpenAI)", "gemini_ai": "gemini (Google)",
                  "whatsapp_ai": "unknown (sent through WhatsApp)"}


def incoming_part(taken: set) -> list[dict]:
    """Pictures only the team can supply. Originals become detector rows;
    WhatsApp copies whose original (same file name) is present become paired
    `robustness` rows (variant_of the original), so a copy is never counted as
    a second independent picture. run_exam.py reports all of them in their own
    section, outside the headline."""
    rows, by_folder = [], {}
    for folder, (label, note) in INCOMING.items():
        d = EXAM_DIR / "incoming" / folder
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "WHAT_GOES_HERE.txt"
        readme.write_text(f"{note}\nLabel given to everything in this folder: {label}.\n"
                          "Then run:  python evaluation/build_exam_set.py --incoming-only\n", encoding="utf-8")
        by_folder[folder] = {}
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".heic", ".heif"):
                print(f"  [skip] {folder}/{p.name}: HEIC can't be read here; re-take or export it as JPEG "
                      "(iPhone: Settings > Camera > Formats > Most Compatible)")
                continue
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            data = p.read_bytes()
            h = sha256(data)
            if h in taken:
                continue
            try:
                w, hgt, fmt = image_facts(data)
            except Exception:  # noqa: BLE001 -- e.g. HEIC without a decoder
                print(f"  [skip] {p.name}: not an image this Python can read")
                continue
            taken.add(h)
            original = next((by_folder[o][p.stem] for o in WHATSAPP_ORIGINALS.get(folder, ())
                             if p.stem in by_folder.get(o, {})), None)
            r = row(id=f"user-{folder}-{p.stem}"[:80], set="exam",
                    purpose="robustness" if original else "detector",
                    file=f"incoming/{folder}/{p.name}", label=label, source=f"user-{folder}",
                    generator=USER_GENERATOR.get(folder, ""),
                    generator_group="real photo" if label == "real" else (
                        "closed commercial" if folder in ("chatgpt_ai", "gemini_ai") else "other / unknown"),
                    content="image", licence="the team's own", credit="supplied by the team",
                    width=w, height=hgt, format=fmt, bytes=len(data), sha256=h,
                    variant_of=original["id"] if original else "",
                    transform="whatsapp round trip" if folder.startswith("whatsapp") else "",
                    notes="" if original or not folder.startswith("whatsapp") else
                    "no original with the same file name was found, so this copy is scored on its own")
            by_folder[folder][p.stem] = r
            rows.append(r)
    return rows


# ------------------------------------------------------------- subjects --

SUBJECTS = {
    "portrait": ["a close-up portrait photo of a person's face", "a headshot of a person"],
    "people": ["a photo of a group of people", "people in a crowd", "a person doing an activity"],
    "animal": ["a photo of an animal", "a pet or a wild animal"],
    "street or buildings": ["a photo of a city street", "buildings and architecture"],
    "nature or landscape": ["a landscape photo", "a nature scene with trees, water or mountains"],
    "indoor scene": ["a photo of a room indoors", "an interior with furniture"],
    "food": ["a photo of food", "a meal on a plate"],
    "object": ["a close-up photo of an object", "a product photo"],
    "vehicle": ["a photo of a car, bus, train or aircraft"],
    "art or illustration": ["a painting", "a digital illustration or anime drawing", "a cartoon"],
    "text or graphic": ["a poster or meme with text", "a screenshot or graphic design"],
}


def label_subjects(rows: list[dict], base: dict) -> None:
    """Fills `subject` using CLIP zero-shot -- the same OpenCLIP weights the
    registry's namer uses. These are machine guesses, used only to group the
    bias map, and the manifest says so."""
    cache_path = DATA / "subject_labels_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for r in rows:
        if r["purpose"] == "detector" and r["sha256"] in cache:
            r["subject"] = cache[r["sha256"]]
    todo = [r for r in rows if r["purpose"] == "detector" and not r["subject"]]
    if not todo:
        return
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="datacomp_xl_s13b_b90k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    with torch.no_grad():
        text = []
        for prompts in SUBJECTS.values():
            e = model.encode_text(tokenizer(prompts))
            e = e / e.norm(dim=-1, keepdim=True)
            text.append(e.mean(0))
        text = torch.stack(text)
        text = text / text.norm(dim=-1, keepdim=True)
        names = list(SUBJECTS)
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            ims = torch.stack([preprocess(Image.open(base[r["set"]] / r["file"]).convert("RGB")) for r in batch])
            f = model.encode_image(ims)
            f = f / f.norm(dim=-1, keepdim=True)
            best = (f @ text.T).argmax(-1).tolist()
            for r, b in zip(batch, best):
                r["subject"] = names[b]
                cache[r["sha256"]] = names[b]
    cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    print(f"  labelled subjects for {len(todo)} images (CLIP zero-shot)")


# ------------------------------------------------------------------ main --

def write_manifest(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["purpose"], r["source"], r["id"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check() -> int:
    bad = 0
    for path, base in ((EXAM_MANIFEST, EXAM_DIR), (CAL_MANIFEST, CAL_DIR)):
        rows = read_manifest(path)
        for r in rows:
            f = base / r["file"]
            if not f.exists():
                print(f"MISSING  {r['id']}  {f}")
                bad += 1
            elif sha256(f.read_bytes()) != r["sha256"]:
                print(f"CHANGED  {r['id']}  {f}")
                bad += 1
        hashes = Counter(r["sha256"] for r in rows)
        dup = [h for h, n in hashes.items() if n > 1]
        if dup:
            print(f"{path.name}: {len(dup)} duplicate image(s)")
            bad += len(dup)
        print(f"{path.name}: {len(rows)} rows checked")
    overlap = {r["sha256"] for r in read_manifest(EXAM_MANIFEST)} & {r["sha256"] for r in read_manifest(CAL_MANIFEST)}
    if overlap:
        print(f"{len(overlap)} image(s) appear in BOTH sets")
        bad += len(overlap)
    print("OK" if not bad else f"{bad} problem(s)")
    return bad


def build() -> None:
    def rng(part: str) -> random.Random:
        # One stream per part: adding or reordering parts never changes
        # which images another part picks.
        return random.Random(f"{SEED}:{part}")

    taken: set = set()
    exam, cal = [], []

    print("Exam set -- OpenFake core/test (generators no training split contains):")
    core_quotas = {g: 10 for g in ("gpt-image-1.5", "gpt-image-2", "nano-banana-pro", "midjourney-7",
                                   "seedream-v5.0", "flux.2-klein-9b", "z-image-turbo", "illustrious",
                                   "sora-2", "veo-3")}
    core_quotas.update({g: 6 for g in ("wan-video-2.5", "recraft-v3", "recraft-v2", "ideogram-2.0",
                                       "ernie-image", "ernie-image-turbo", "lumina-17-2-25",
                                       "aurora-20-1-25", "halfmoon-4-4-25", "frames-23-1-25")})
    core_quotas.update({"real:imagenet": 70, "real:docci": 70})
    exam += openfake_part("exam", EXAM_DIR, "openfake_core", ["core/test-00000-of-00013.parquet",
                                                              "core/test-00006-of-00013.parquet"],
                          lambda r: r["model"] if r["label"] == "fake" else f"real:{r['model']}",
                          core_quotas, 18, rng("openfake_core"), "openfake-core-test", taken)

    print("Exam set -- OpenFake reddit/test (pictures as they circulate online):")
    exam += openfake_part("exam", EXAM_DIR, "openfake_reddit", ["reddit/test-00000-of-00005.parquet",
                                                                "reddit/test-00003-of-00005.parquet"],
                          lambda r: f"{r['label']}:{r['type']}",
                          {"real:image": 50, "fake:image": 50, "real:video": 15, "fake:video": 15},
                          10, rng("openfake_reddit"), "openfake-reddit-test", taken)

    used_members: set = set()  # zip members already taken, so calibration never repeats one
    print("Exam set -- older generators (CNNDetection):")
    exam += zip_part("exam", EXAM_DIR, "cnndetection", f"{CNNDET}/CNN_synth_testset.zip", {
        ("stylegan2/", "ai"): 8, ("progan/", "ai"): 8, ("biggan/", "ai"): 8, ("cyclegan/", "ai"): 8,
        ("stargan/", "ai"): 6, ("gaugan/", "ai"): 6, ("whichfaceisreal/", "ai"): 6,
        ("stylegan2/", "real"): 6, ("progan/", "real"): 6, ("biggan/", "real"): 6,
        ("cyclegan/", "real"): 6, ("whichfaceisreal/", "real"): 6,
    }, "cnndetection", rng("cnndetection_exam"), taken, used_members)
    print("Exam set -- older generators (GenImage):")
    exam += zip_part("exam", EXAM_DIR, "genimage", f"{GENIMAGE}/genimage_test.zip", {
        ("test/glide_imagenet/ai/", "ai"): 8, ("test/sdv4_imagenet/ai/", "ai"): 6,
        ("test/sdv5_imagenet/ai/", "ai"): 6, ("test/adm_imagenet/ai/", "ai"): 6,
        ("test/vqdm_imagenet/ai/", "ai"): 6, ("test/wukong_imagenet/ai/", "ai"): 6,
        ("test/midjourney_imagenet/ai/", "ai"): 6, ("test/biggan_imagenet/ai/", "ai"): 4,
        ("test/sdv5_imagenet/nature/", "real"): 8, ("test/glide_imagenet/nature/", "real"): 8,
    }, "genimage", rng("genimage_exam"), taken, used_members)

    print("Exam set -- this project's own test images and Commons:")
    exam += local_part(taken)
    print("Exam set -- C2PA public test files (for Layer 0):")
    exam += c2pa_part(taken)
    print("Exam set -- watermarked samples (for Layer 1):")
    exam += watermark_part(exam, rng("watermark"), taken)
    print("Exam set -- simulated social-media copies:")
    exam += social_part(exam, rng("social"), taken)
    exam += incoming_part(taken)

    print("Calibration set -- OpenFake core/validation (a different split from the exam):")
    val_index = openfake_index("core/validation-00000-of-00015.parquet")
    groups = sorted({r["row_group"] for r in val_index})
    val_rng = rng("openfake_validation")
    val_rng.shuffle(groups)
    chosen = set(groups[:5])
    val_quota = {"real:laion": 60, "real:pexels": 60}
    val_quota.update({m: 3 for m in sorted({r["model"] for r in val_index
                                            if r["label"] == "fake" and r["row_group"] in chosen})})
    cal += openfake_part("calibration", CAL_DIR, "openfake_validation", ["core/validation-00000-of-00015.parquet"],
                         lambda r: r["model"] if r["label"] == "fake" else f"real:{r['model']}",
                         val_quota, 5, val_rng, "openfake-core-validation", taken)
    print("Calibration set -- the source panel's own 8 generators, new samples:")
    cal += zip_part("calibration", CAL_DIR, "cnndetection", f"{CNNDET}/CNN_synth_testset.zip", {
        ("stylegan2/", "ai"): 10, ("progan/", "ai"): 10, ("biggan/", "ai"): 10, ("cyclegan/", "ai"): 10,
        ("stylegan2/", "real"): 5, ("progan/", "real"): 5, ("biggan/", "real"): 5, ("cyclegan/", "real"): 5,
    }, "cnndetection", rng("cnndetection_cal"), taken, used_members)
    cal += zip_part("calibration", CAL_DIR, "genimage", f"{GENIMAGE}/genimage_test.zip", {
        ("test/glide_imagenet/ai/", "ai"): 10, ("test/sdv4_imagenet/ai/", "ai"): 5,
        ("test/sdv5_imagenet/ai/", "ai"): 5, ("test/biggan_imagenet/ai/", "ai"): 5,
        ("test/glide_imagenet/nature/", "real"): 5, ("test/sdv4_imagenet/nature/", "real"): 5,
    }, "genimage", rng("genimage_cal"), taken, used_members)

    for name, rows in (("exam", exam), ("calibration", cal)):
        dup = [i for i, n in Counter(r["id"] for r in rows).items() if n > 1]
        if dup:
            raise SystemExit(f"{name}: duplicate ids {dup[:5]} -- fix the id scheme before writing")

    label_subjects(exam + cal, {"exam": EXAM_DIR, "calibration": CAL_DIR})
    subj = {r["id"]: r["subject"] for r in exam}
    for r in exam:  # derived copies take their original's subject
        if r["variant_of"]:
            r["subject"] = subj.get(r["variant_of"], r["subject"])

    write_manifest(EXAM_MANIFEST, exam)
    write_manifest(CAL_MANIFEST, cal)
    for name, rows in (("exam", exam), ("calibration", cal)):
        c = Counter((r["purpose"], r["label"] or "-") for r in rows)
        print(f"{name}: {len(rows)} rows  " + ", ".join(f"{p}/{l}={n}" for (p, l), n in sorted(c.items())))


def incoming_only() -> None:
    """Adds (or refreshes) the team's own pictures in the exam manifest without
    rebuilding anything else -- no network needed."""
    rows = [r for r in read_manifest(EXAM_MANIFEST) if not r["source"].startswith("user-")]
    taken = {r["sha256"] for r in rows} | {r["sha256"] for r in read_manifest(CAL_MANIFEST)}
    new = incoming_part(taken)
    label_subjects(new, {"exam": EXAM_DIR})
    write_manifest(EXAM_MANIFEST, rows + new)
    c = Counter((r["source"], r["purpose"]) for r in new)
    print(f"{len(new)} picture(s) from incoming/: " + (", ".join(f"{s}/{p}={n}" for (s, p), n in sorted(c.items())) or "none"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or check the Deep-Guard exam and calibration sets.")
    parser.add_argument("--check", action="store_true", help="verify every file against the manifests")
    parser.add_argument("--incoming-only", action="store_true",
                        help="only add the pictures in incoming/ (offline; everything else is left as it is)")
    args = parser.parse_args()
    if args.check:
        sys.exit(1 if check() else 0)
    incoming_only() if args.incoming_only else build()
    sys.exit(1 if check() else 0)


if __name__ == "__main__":
    main()
