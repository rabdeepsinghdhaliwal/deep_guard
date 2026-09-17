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

## Quick start

Requires Python 3.10+. All three model files below are already
included in `models/` — you do not need to train or download anything
to run the server.

```bash
cd deepguard-bouncer
python -m venv venv
```

Activate it (pick the line for your shell):

```bash
venv\Scripts\activate          # Windows (cmd/PowerShell)
source venv/bin/activate        # macOS/Linux
```

Install dependencies and start the server:

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000**. The status pill at the top right
should read "Ready" with a green dot within a few seconds — that means
the base detector loaded correctly. If it reads "Model unavailable,"
see **Troubleshooting** below.

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
photography. It was found by testing, not assumed — see `bias_map/`
below for how that kind of gap gets caught systematically instead of
by hand.

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

**Idea 1 — Bias Map** (`bias_map/`, `GET /bias-map`, `GET
/api/bias-map`). One aggregate accuracy number can hide a subgroup the
model does badly on — exactly what happened here: an early 98.17%
score was 91.5% one easy category and 8.5% one hard one. This slices
accuracy by generator/subject/style and renders it as a grid instead
of one number. Run `bias_map/build_bias_map.py` by hand to regenerate
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
`demo_artworks/`.

**Idea 2 — Generator Attribution** (`attribution.py`, `model.py`'s
`GENERATOR_CLASSES`). Once an image is flagged as AI-generated, a
*separate* model (trained independently from the Stage 1 detector)
estimates which of 8 specific tools most likely made it — StyleGAN2,
ProGAN, BigGAN, CycleGAN, DDPM, Latent Diffusion, Stable Diffusion, or
GLIDE — since every generator leaves faint, characteristic statistical
traces. Trained on a real subset of the ArtiFact dataset (Kaggle);
current checkpoint scores 81.85% on its own held-out test split.

> **Honest limitation:** that accuracy is against generators the model
> *trained on*. It has not been tested against a generator it never
> saw (e.g. MidJourney, Flux) — a known hard problem in this research
> area (GAN fingerprints often don't transfer to diffusion models,
> and vice versa). That generalization test is the natural next step,
> not yet done.

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
│   ├── content_registry.py     # Idea 4 — SSCD embeddings + FAISS search
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
└── models/                     # See PUT_WEIGHTS_HERE.md — all three files included
```

---

## Running the tests

```bash
cd app
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

47 tests: unit tests for the pure-logic modules (fingerprint hashing/
signing, frequency-domain peak detection, model architecture shapes)
plus integration tests against a real FastAPI `TestClient` with real
models loaded — every API endpoint, across all 8 real labeled test
images. Two of them pin, by name, bugs that were actually found and
fixed this project (a flat-image edge case in the frequency panel, an
oversized-body DoS on `/api/fingerprint/verify`) so those exact
regressions can't silently come back. The Content Registry's
persisted files are redirected to a temp directory for the test run —
running the suite never touches the real `models/content_registry.*`
files. Takes about 15 seconds, dominated by loading ~130MB of real
model weights once at session start.

---

## Retraining

Both notebooks in `colab_notebook/` follow the same disciplined
two-phase approach (frozen backbone warm-up, then full fine-tune with
a proper train/val/test split) and are meant to run on Google Colab
with a free GPU:

1. Open the notebook in Colab, enable a GPU runtime (**Runtime → Change
   runtime type → T4 GPU**), and **Runtime → Run all**.
2. Partway through, upload your Kaggle API credentials (`kaggle.json`,
   from kaggle.com → Settings → API → Create New Token) when prompted
   — needed to download the training dataset.
3. At the end, the notebook downloads a `.pth` checkpoint. Move it into
   `models/`, replacing the existing file with the same name.
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
`models/deepguard_bouncer.pth` — see `models/PUT_WEIGHTS_HERE.md`.

**"Couldn't reach the analysis service."** The `uvicorn` process
stopped or crashed — check its terminal for a traceback.

**Content Registry pages return "unavailable" / 503.** Either
`faiss-cpu` didn't install (check `pip install -r requirements.txt`
output for errors) or `models/sscd_disc_mixup.torchscript.pt` is
missing. Both the base detector and the attribution model still work
fine regardless — this feature degrades independently, by design.

**"Likely source" panel never appears.** It only shows once an image
is confidently flagged as AI-generated (server-side ≥50% *and* the
displayed verdict is "manipulated," a stricter client-side bar) —
requires `models/deepguard_attribution.pth` to exist at all.

---

## What's not done yet

Kept here deliberately instead of hidden, since an honest limitations
list is worth more than pretending everything is finished:

- **Bias Map's dataset is a placeholder** (8 images, no real generator
  labels) — the tool works, the data behind it isn't trustworthy yet.
- **Generator Attribution's cross-generator generalization is
  untested** — only measured against generators it trained on.
- **Live/webcam temporal detection** was scoped out of this phase
  entirely, not attempted.
