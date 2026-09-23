# Deep-Guard

A local web app that answers three separate questions about a photo or
video someone shows you: **is this AI-generated**, **has this exact
file been tampered with since it was checked**, and **is this a crop,
edit, or reuse of something already registered**. All three run
locally — nothing you upload is stored or sent anywhere.

This document covers the whole project as it stands today. If you only
want to run it, skip to **Quick start**. If you want the reasoning
behind each feature, skip to **What's actually in here**.

---

## Quick start (on a new machine)

Requires git and Python 3.10 or newer (developed on Windows 11 with
Python 3.13). No GPU needed — everything runs on the CPU.

**1. Get the code.** The three model files the app needs are part of the
repository, so cloning downloads them (about 125 MB). There is nothing
to train and no separate model download.

```bash
git clone https://github.com/rabdeepsinghdhaliwal/deep_guard.git
cd deep_guard/deepguard-bouncer
python -m venv venv
```

**2. Activate the virtual environment** (pick the line for your shell):

```bash
venv\Scripts\activate          # Windows (cmd/PowerShell)
source venv/bin/activate        # macOS/Linux
```

**3. Install the dependencies.**

```bash
cd app
pip install -r requirements.txt
```

On Linux, a plain `pip install torch` fetches the multi-gigabyte CUDA
build. The app only uses the CPU, so there run this first, then the
line above:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**4. Optional, once per machine: build the registry's "namer".** The
Content Registry names each area of an image ("face", "landscape") with
a CLIP model that is too big for GitHub, so it is built locally. It
needs internet: it downloads 605 MB from Hugging Face (into Hugging
Face's own cache, not this repo) and writes a 176 MB file to `deepguard-bouncer/models/`,
which git ignores. Skip it and everything still works — areas are just
named by position ("top left area") instead of by content.

```bash
pip install -r requirements-dev.txt
python ../tools/build_concept_labeler.py
```

**5. Start the server.** (If you build the namer later, restart the
server so it picks it up.)

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000**. The status pill at the top right
should read "Ready" with a green dot within a few seconds — that means
the base detector loaded correctly. If it reads "Model unavailable,"
see **Troubleshooting** below.

### What you need, and where it comes from

| What | Where it comes from |
|---|---|
| Detector, attribution and SSCD models (`deepguard-bouncer/models/*.pth`, `deepguard-bouncer/models/sscd_disc_mixup.torchscript.pt`) | In the repo — `git clone` downloads them |
| Demo paintings, the 8 test photos, the 44 generalization-test images | In the repo |
| Python packages | `pip install -r requirements.txt` (plus `requirements-dev.txt` for the tests and the namer build) |
| Registry namer (`deepguard-bouncer/models/concept_labeler_*`) | Optional — built once by `deepguard-bouncer/tools/build_concept_labeler.py` (step 4) |
| Signing key pair (`deepguard-bouncer/models/demo_signer_*.pem`) | Created by the server on first start — every installation signs with its own key |
| Registry index, records and stored points (`deepguard-bouncer/models/content_registry*`) | Created on first start, seeded with the three demo paintings |

Because every installation gets its own signing key, after your first
start git shows `deepguard-bouncer/models/demo_signer_public.pem` as changed. That file is
now *your* server's public key — leave it changed, but **don't commit
it**: it would replace the key the original server's signatures are
checked against. Your private key is gitignored and never leaves
`deepguard-bouncer/models/`.

Three other pages exist off the same server:

| Page | What it's for |
|---|---|
| `/` | Upload a photo/video, get the AI-verdict + fingerprint + explainability panels |
| `/bias-map` | See the detector's accuracy sliced by category instead of one aggregate number |
| `/registry` | Register an image's content, or check whether an upload is a crop/edit of something already registered |

---

## What's actually in here

Deep-Guard was built in stages. Each one is real, working code today —
nothing described below is a mockup.

### Stage 1 — the detector

`model.py` fine-tunes an EfficientNet-B0 (transfer learning, two-phase:
frozen backbone warm-up, then full unfreeze) to output one number:
P(this image/video is AI-generated). `POST /api/analyze` accepts an
image or video; videos are sampled across their full duration (not
just one frame) and reported as a timeline, so manipulation confined
to part of a clip still shows up instead of being averaged away.

**Known limitation, disclosed on the result page itself:** this model
under-detects AI images styled like polished, professional stock
photography. It was found by testing, not assumed — see `deepguard-bouncer/bias_map/`
below for how that kind of gap gets caught systematically instead of
by hand. The generalization test below (`generalization_test/`)
independently reproduced the same weakness on a different, real-world
sample: 30 real DALL·E 3 images were flagged at the app's real
80%-confidence threshold 100% of the time, but only 35.7% of real
MidJourney images were — MidJourney's more photorealistic house style
appears to hit the same blind spot.

### Stage 2 — the authenticity fingerprint

`fingerprint.py` computes, for every analyzed file: an exact SHA-256
hash (changes completely if even one byte changes) and a perceptual
hash (survives re-compression — the WhatsApp/Twitter problem, where
lossy compression shouldn't make a genuine photo look "tampered").
Both are signed with the server's own Ed25519 private key. Two
endpoints make this provable rather than just asserted:

- `POST /api/fingerprint/compression-test` — re-saves your upload at
  several JPEG qualities server-side and shows both fingerprints
  changing (or not) at each quality, live.
- `POST /api/fingerprint/verify` — independently re-verifies a
  fingerprint record's signature against the server's public key,
  offline logic anyone could run themselves. Flip one character of a
  hash and verification correctly fails.

### Phase 2 — five extensions, each addressing a specific gap

**Idea 1 — Bias Map** (`deepguard-bouncer/bias_map/`, `GET /bias-map`, `GET
/api/bias-map`). One aggregate accuracy number can hide a subgroup the
model does badly on — exactly what happened here: an early 98.17%
score was 91.5% one easy category and 8.5% one hard one. This slices
accuracy by generator/subject/style and renders it as a grid instead
of one number. Run `deepguard-bouncer/bias_map/build_bias_map.py` by hand to regenerate
`results.json` against a new labeled test set; the endpoint only
reads that file.

> **Honest limitation:** the shipped `results.json` was generated on
> 8 images with no real generator labels — enough to prove the
> pipeline works end-to-end, not enough to trust the numbers. The page
> says this explicitly rather than hiding it. A real bias map needs a
> properly labeled 300–500 image set across generator × subject ×
> style.

**Idea 5a/5b — Quantified explainability** (`shap_explain.py`,
`frequency_analysis.py`, folded into `/api/analyze`'s
`explainability` field). Grad-CAM shows *where* the model looked as a
heatmap; these answer the two follow-ups a heatmap can't. SHAP
(`POST /api/explain/shap-regions`) breaks the verdict down by image
region into signed numbers ("this region contributed +0.31 toward
fake") using Shapley values over superpixel segments. The frequency
panel computes a Fourier spectrum of the actual pixels — no model
involved — since real cameras and AI generators leave measurably
different traces in an image's frequency domain.

**Idea 4 — Content Registry** (`content_registry.py`, `/registry`,
`POST /api/registry/register`, `POST /api/registry/check`). The
fingerprint above proves "is this the exact same file" — a crop
breaks that almost immediately, since a perceptual hash describes the
whole frame. This answers a harder question: *is this a crop, edit,
or partial reuse of something already registered?* — the mechanism
behind catching someone who crops a signature off a registered
artwork and resells it. Uses Meta AI's pretrained **SSCD** model
(`sscd_disc_mixup.torchscript.pt`) to embed image content into a
vector that survives cropping/rotation/recolouring, searched via
**FAISS**. Seeded on first run with three public-domain artworks from
`deepguard-bouncer/demo_artworks/`.

**The registry explains itself** (`registry_explain.py`,
`local_features.py`, `concept_labels.py`). SSCD's 512 numbers decide
*whether* something is a copy, but none of them individually means
anything a person could read — so on their own they can't say what two
images have in common. Every upload now gets its **distinct features**,
and every match a **"why it matched"**. Three measured layers, the same
way a fingerprint examiner backs up "these match" by marking the ridge
points both prints share:

1. **Distinctive points** — SIFT keypoints (Lowe, *Distinctive Image
   Features from Scale-Invariant Keypoints*, IJCV 2004), compared as
   RootSIFT, paired with Lowe's ratio test and verified with RANSAC:
   only points that agree on one placement of the copy inside the
   original count as shared. The placement itself says what was done to
   the copy: how much of the original it shows and from where, rotation,
   mirroring, size, added borders — and, comparing an 8×8 colour grid
   once aligned, whether colour was removed, changed or brightened.
2. **Named areas** — at registration the original is split 3×3 and a
   CLIP model (OpenCLIP ViT-B/32, DataComp-XL weights, MIT licence; image
   half only, vocabulary pre-encoded by `deepguard-bouncer/tools/build_concept_labeler.py`)
   names each area. This gives the list of *things* ("face", "hands",
   "landscape"). Names are a model's guess and are shown as one.
3. **Responsibility** — SSCD's own score is split exactly among those
   named things with Shapley values over its grid of feature cells
   (9×14 for the Mona Lisa; GeM pooling makes every subset of cells computable without another
   network pass). The shares add back up to the reported similarity, so
   "what was responsible for the detection" is answered by the network
   that made the detection. Content that isn't from the original — an
   added border, rotation padding — gets a negative share.

Measured (`registry_explanations_test/run_measurements.py`, results
in its `results.json`): **all 24 real copies** tested (18 edits of the
Mona Lisa — crops from 80% down to 25%, rotations, a mirror, black and
white, sepia, brightening, blur, heavy compression, text overlay, an
added border — plus 6 colour edits of the Great Wave) were confirmed by 61 to
1,504 shared points, **with every fact correct** — crop area, position,
rotation (15.0° and 90.0°), mirroring, black-and-white, sepia,
brightening. Across **594 pairs of unrelated images** (paintings and
test photos against each other and against the 44 MidJourney/DALL·E 3
images) the most points ever shared by chance was **9** (mean 1.4), so
"confirmed" is set at 20. The match decision itself is unchanged —
SSCD at 0.4 still decides; explanations never add or remove a match.
The registry still never keeps the image: a registration now also keeps
its points' positions and descriptions, the area names and an 8×8 grid
of average colours (about 100–200 KB).

**Idea 2 — Generator Attribution** (`attribution.py`, `model.py`'s
`GENERATOR_CLASSES`). Once an image is flagged as AI-generated, a
*separate* model (trained independently from the Stage 1 detector)
estimates which of 8 specific tools most likely made it — StyleGAN2,
ProGAN, BigGAN, CycleGAN, DDPM, Latent Diffusion, Stable Diffusion, or
GLIDE — since every generator leaves faint, characteristic statistical
traces. Trained on a real subset of the ArtiFact dataset (Kaggle);
current checkpoint scores 81.85% on its own held-out test split.

> **Honest limitation, now measured (see `generalization_test/`):**
> tested against 44 real images from two generators it never trained
> on — 14 from MidJourney's public showcase, 30 from OpenAI's own
> official DALL·E 3 examples page, both first-party sources. The model
> cannot say "I don't know" (it's a fixed 8-class softmax), so every
> prediction on these is wrong by construction — what matters is *how*
> it's wrong. The result is mixed: when wrong, it's usually
> *coherently* wrong (90.9% of the time it lands on another diffusion-
> family tool rather than a GAN, most often defaulting to "Stable
> Diffusion") — a real, if coarse, transferable signal, not noise. But
> it is **not** appropriately less confident on unfamiliar input: mean
> confidence on these wrong guesses (74.2%) is about as high as on
> data it actually trained on, so the confidence number alone gives no
> warning the panel might be unreliable. The live "Likely source"
> panel's own copy now discloses this directly.

**Idea 3 — Live temporal (webcam) detection** was deliberately scoped
*out* of this phase — it needs real-time video infrastructure this
project doesn't have yet. Not started, not attempted.

---

## Project structure

```
deepguard-bouncer/
├── app/
│   ├── main.py                # FastAPI app — every endpoint, see its module docstring
│   ├── model.py                # EfficientNet-B0 architectures (detector + attribution)
│   ├── fingerprint.py          # SHA-256 + perceptual hash + Ed25519 signing (Stage 2)
│   ├── gradcam.py              # Heatmap explainability
│   ├── uncertainty.py          # Monte Carlo Dropout consistency check
│   ├── frequency_analysis.py   # Idea 5b — Fourier spectrum panel
│   ├── shap_explain.py         # Idea 5a — Shapley-value region attribution
│   ├── content_registry.py     # Idea 4 — SSCD embeddings + FAISS search (+ exact Shapley split of its score)
│   ├── local_features.py       # Registry explanations — SIFT points, RANSAC placement, colour comparison
│   ├── concept_labels.py       # Registry explanations — CLIP "namer" (optional, image half only)
│   ├── registry_explain.py     # Registry explanations — distinct features + "why it matched"
│   ├── attribution.py          # Idea 2 — generator attribution inference
│   ├── requirements.txt
│   └── static/                 # Plain HTML/CSS/JS frontend, no build step
├── bias_map/
│   ├── build_bias_map.py       # Standalone script — run by hand, not by a request
│   ├── labeled_test_set.csv    # filename, generator, subject, style
│   └── results.json            # Written by the script above; read by /api/bias-map
├── colab_notebook/
│   ├── DeepGuard_Training_MASSIVE.ipynb        # Trains deepguard_bouncer.pth
│   └── DeepGuard_Attribution_Training.ipynb    # Trains deepguard_attribution.pth
├── demo_artworks/              # Seed images for the Content Registry
├── tools/
│   └── build_concept_labeler.py   # One-time build of the registry's CLIP namer (optional)
└── models/                     # See PUT_WEIGHTS_HERE.md — the three core files included
```

At the repo root, next to `deepguard-bouncer/`: `test_images/` (the 8
labelled demo photos), `generalization_test/` (attribution on unseen
generators) and `registry_explanations_test/` (the measurements behind
the registry's explanations).

---

## Running the tests

```bash
cd deepguard-bouncer/app
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

89 tests: unit tests for the pure-logic modules (fingerprint hashing/
signing, frequency-domain peak detection, model architecture shapes,
distinctive-point matching) plus integration tests against a real
FastAPI `TestClient` with real models loaded — every API endpoint,
across all 8 real labeled test images. Two of them pin, by name, bugs
that were actually found and fixed this project (a flat-image edge
case in the frequency panel, an oversized-body DoS on
`/api/fingerprint/verify`) so those exact regressions can't silently
come back. The registry explanations are pinned by a parametrised
table of 20 real edits that must each be confirmed *and* described
correctly, by the exact-split property (shares add up to the score),
and by the graceful paths (no namer, an entry registered before
explanations existed, a stale features file). The Content Registry's
persisted files are redirected to a temp directory for the test run —
running the suite never touches the real `deepguard-bouncer/models/content_registry*`
files. Takes about 40 seconds, dominated by loading the real model
weights once at session start.

---

## Retraining

Both notebooks in `deepguard-bouncer/colab_notebook/` follow the same disciplined
two-phase approach (frozen backbone warm-up, then full fine-tune with
a proper train/val/test split) and are meant to run on Google Colab
with a free GPU:

1. Open the notebook in Colab, enable a GPU runtime (**Runtime → Change
   runtime type → T4 GPU**), and **Runtime → Run all**.
2. Partway through, upload your Kaggle API credentials (`kaggle.json`,
   from kaggle.com → Settings → API → Create New Token) when prompted
   — needed to download the training dataset.
3. At the end, the notebook downloads a `.pth` checkpoint. Move it into
   `deepguard-bouncer/models/`, replacing the existing file with the same name.
4. Restart the server.

`DeepGuard_Training_MASSIVE.ipynb` produces `deepguard_bouncer.pth`
(the real/AI detector). `DeepGuard_Attribution_Training.ipynb` produces
`deepguard_attribution.pth` (the generator-attribution model, trained
on the ArtiFact dataset). Both notebooks assert their own data-split
integrity as they run — a red error box means something needs fixing
before the next cell will succeed, not something to skip past.

---

## Troubleshooting

**Status pill shows "Model unavailable."** The server can't find
`deepguard-bouncer/models/deepguard_bouncer.pth` — see `deepguard-bouncer/models/PUT_WEIGHTS_HERE.md`.

**"Couldn't reach the analysis service."** The `uvicorn` process
stopped or crashed — check its terminal for a traceback.

**Content Registry pages return "unavailable" / 503.** Either
`faiss-cpu` didn't install (check `pip install -r requirements.txt`
output for errors) or `deepguard-bouncer/models/sscd_disc_mixup.torchscript.pt` is
missing. Both the base detector and the attribution model still work
fine regardless — this feature degrades independently, by design.

**"Likely source" panel never appears.** It only shows once an image
is confidently flagged as AI-generated (server-side ≥50% *and* the
displayed verdict is "manipulated," a stricter client-side bar) —
requires `deepguard-bouncer/models/deepguard_attribution.pth` to exist at all.

**Registry areas are named "top left area" etc. instead of "face".**
The optional namer isn't built on this machine — see the end of
**Quick start** (`deepguard-bouncer/tools/build_concept_labeler.py`).

**A match says "explanation unavailable".** That entry was registered
before the registry kept distinctive points, and since the registry
never keeps images there is nothing to rebuild them from. Register the
image again. (The three demo artworks are rebuilt automatically on
startup, since their files are in `deepguard-bouncer/demo_artworks/`.) To reset the
registry completely, delete `deepguard-bouncer/models/content_registry.index`,
`deepguard-bouncer/models/content_registry_metadata.json` and the
`deepguard-bouncer/models/content_registry_features/` folder.

---

## What's not done yet

Kept here deliberately instead of hidden, since an honest limitations
list is worth more than pretending everything is finished:

- **Bias Map's dataset is a placeholder** (8 images, no real generator
  labels) — the tool works, the data behind it isn't trustworthy yet.
- **Generator Attribution's confidence doesn't drop on unseen
  generators** — measured (see `generalization_test/`), not a guess:
  it's family-coherent 91% of the time but not appropriately less
  confident, so the confidence number alone can't warn a user the
  panel is guessing outside its training set.
- **Live/webcam temporal detection** was scoped out of this phase
  entirely, not attempted.
- **The registry still misses a 25% fragment** (SSCD scores it 0.285,
  under the 0.4 line) even though it shares 61 distinctive points with
  the original — far above the 9 that chance ever produced. Using
  points to rescue near-misses would catch it, but it changes which
  images match, so it's a separate, deliberate decision, not part of
  the explanations.
- **Registry area names are CLIP's guesses** and can be wrong,
  especially on black-and-white copies and close-up faces (a 3×3 area
  of a face is often just "face" or "person"). The 3×3 grid can also
  cut one object into two areas. The points and the score split are
  measured; the names are not evidence.
- **Stored distinctive points are sensitive data.** The registry keeps
  no pixels, but research has reconstructed rough pictures from SIFT
  descriptors (Weinzaepfel et al., CVPR 2011).
