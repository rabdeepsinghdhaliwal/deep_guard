# Deep-Guard

A local web app that checks a photo or video someone shows you, and shows
its work. It asks four questions, in this order:

1. **Where did it come from?** If its creator registered it, a signed,
   tamper-evident ledger proves who registered it and when — even after
   WhatsApp has re-saved it or someone has cropped it (Stages 3 and 4).
2. **Has this exact file been changed?** Exact and look-alike
   fingerprints (Stage 2).
3. **Is it a copy of something registered?** Content matching that
   survives crops, flips and colour changes (Stage 2).
4. **Is it AI-generated?** A detector with heatmaps and other
   explanations (Stage 1) — always labelled a guess, because a detector
   can only guess.

*Provenance first, prediction last:* the detector's guess is the fallback
for a picture nobody registered, not the headline.

```
 a photo is made ──► its creator registers it ──► fingerprints + a hidden ID in the pixels
                                                  + Content Credentials (the "CR" label)
                              │
                              ▼
        sealed in a signed block ──► co-signed by witnesses ──► anchored in Bitcoin
                              │
                              ▼
 anyone checks a copy: the Ledger page · the Explorer, in their own browser ·
 a right-click with the Lens extension · their phone, through the public link
                              │
                              ▼
 no record anywhere? ──► the AI detector's guess, with its explanations
```

Everything runs on your own laptop. Pictures are never stored, only the
fingerprints of registered ones. The internet is used only when you ask
for it: to anchor a block's hash in Bitcoin, or to open a public link.

This document covers the whole project as it stands today. If you only
want to run it, skip to **Quick start**. If you want the reasoning
behind each feature, skip to **What's actually in here**.

**Showing it to someone?** [`docs/demo-runbook.html`](docs/demo-runbook.html)
is the demo manual: why each part exists, how to set everything up on a
Windows laptop, and a tested script for every act, from the detector to the
QR code. GitHub shows it as source, so download it (or open it from your
clone) and double-click it to read it in any browser, with no internet needed.

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

**3. Install the dependencies.** The second line adds the ledger's own
few packages: Bitcoin timestamps, the hidden ID, Content Credentials and
the QR code.

```bash
cd app
pip install -r requirements.txt
pip install -r ../../deepguard-ledger/requirements.txt
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

**5. Start Deep-Guard.** The ledger's launcher starts the server and its
two witnesses together. (If you build the namer later, restart it so it
picks the namer up.)

```bash
cd ../../deepguard-ledger
python run_deepguard.py
```

Open **http://127.0.0.1:8000**. The status pill at the top right
should read "Ready" with a green dot within a few seconds — that means
the base detector loaded correctly. If it reads "Model unavailable,"
see **Troubleshooting** below. The very first start also downloads the
hidden-ID model (about 65 MB, from Adobe's server); the server window
says `Hidden-ID model (TrustMark Q) ready` when it has. `Ctrl+C` stops
everything.

To let anyone try it from their phone, start it with
`python run_deepguard.py --public` instead: it prints a public link and a
QR code (this needs Cloudflare's free `cloudflared` program; see
[the ledger README](deepguard-ledger/README.md#a-live-public-link-stage-4)).
The detector on its own, without the ledger, still starts the old way:
`uvicorn main:app --reload` inside `deepguard-bouncer/app`.

### What you need, and where it comes from

| What | Where it comes from |
|---|---|
| Detector, attribution and SSCD models (`deepguard-bouncer/models/*.pth`, `deepguard-bouncer/models/sscd_disc_mixup.torchscript.pt`) | In the repo — `git clone` downloads them |
| Demo paintings, the 8 test photos, the 44 generalization-test images | In the repo |
| Python packages | `pip install -r requirements.txt` (plus `requirements-dev.txt` for the tests and the namer build) |
| Registry namer (`deepguard-bouncer/models/concept_labeler_*`) | Optional — built once by `deepguard-bouncer/tools/build_concept_labeler.py` (step 4) |
| Signing key pair (`deepguard-bouncer/models/demo_signer_*.pem`) | Created by the server on first start — every installation signs with its own key |
| Registry index, records and stored points (`deepguard-bouncer/models/content_registry*`) | Created on first start, seeded with the three demo paintings |
| The ledger: blocks, witness notebooks, creator keys, anchors (`deepguard-ledger/ledger_data/`) | Created on first start by `run_deepguard.py`; gitignored (it holds private keys) |
| Hidden-ID model weights (TrustMark, about 65 MB) | Downloaded from Adobe's server into the `trustmark` package on the first start |
| Content Credentials certificate authority (`deepguard-ledger/credentials/`) | Created on first use — every installation has its own; gitignored |
| C2PA trust lists (`deepguard-ledger/trust/`) | In the repo, pinned; `tools/update_trust_lists.py` refreshes them |
| `cloudflared` (only for `--public`) | Optional — `winget install --id Cloudflare.cloudflared`, or saved as `deepguard-ledger/tools/cloudflared.exe` (gitignored) |

Because every installation gets its own signing key, after your first
start git shows `deepguard-bouncer/models/demo_signer_public.pem` as changed. That file is
now *your* server's public key — leave it changed, but **don't commit
it**: it would replace the key the original server's signatures are
checked against. Your private key is gitignored and never leaves
`deepguard-bouncer/models/`.

The pages, all off the same server:

| Page | What it's for |
|---|---|
| `/` | Upload a photo/video, get the AI-verdict + fingerprint + explainability panels, and (under every result) what the ledger knows about it |
| `/ledger` | Register a file under your name; verify any copy of it (four verdicts); revoke, dispute, resolve |
| `/explorer` | The whole chain drawn and re-checked in your own browser: witnesses, Bitcoin anchors, the tamper lab, receipts |
| `/share` | With `--public`: the public link as a big QR code, for a projector |
| `/bias-map` | See the detector's accuracy sliced by category instead of one aggregate number |
| `/registry` | Register an image's content, or check whether an upload is a crop/edit of something already registered |

The **Deep-Guard Lens** browser extension (`deepguard-lens/`) adds a
right-click *Check with Deep-Guard* to every picture on every website;
installing it takes a minute ([its README](deepguard-lens/README.md)).

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
endpoints and a standalone script make this provable rather than just
asserted:

- `POST /api/fingerprint/compression-test` — re-saves your upload at
  several JPEG qualities server-side and shows both fingerprints
  changing (or not) at each quality, live.
- `POST /api/fingerprint/verify` — re-verifies a fingerprint record's
  signature against the server's public key. Flip one character of a
  hash and verification correctly fails. This is the server checking
  its own seal, which demonstrates the mechanism but isn't independent.
- `deepguard-bouncer/tools/verify_offline.py` — the same check with no
  server at all, the way an outside party would do it: it needs only the
  `cryptography` package, a saved record and the signer's public key
  from a source you trust (for your own server,
  `deepguard-bouncer/models/demo_signer_public.pem`, also shown in the
  page footer), never the key inside the record. It takes a bare record
  or a server answer saved as-is from `/api/analyze`,
  `/api/registry/register` or `/api/registry/check`:

  ```bash
  cd deepguard-bouncer
  curl -F "file=@photo.jpg" http://127.0.0.1:8000/api/analyze -o saved.json
  python tools/verify_offline.py saved.json models/demo_signer_public.pem
  ```

  (In Windows PowerShell, type `curl.exe` instead of `curl`.) Exit code 0
  means every record is valid, 1 that one isn't, 2 that it couldn't
  check. A valid seal proves the record is genuine and unchanged, not
  that the picture is true or who created it.

### Phase 2 — five extensions, each addressing a specific gap

**Idea 1 — Bias Map** (`deepguard-bouncer/bias_map/`, `GET /bias-map`, `GET
/api/bias-map`). One aggregate accuracy number can hide a subgroup the
model does badly on — exactly what happened here: an early 98.17%
score was 91.5% one easy category and 8.5% one hard one. This slices
the analysis page's results by generator, source, subject, image size
and file format and renders them as a grid instead of one number.
Since Step 0 it is computed on the exam set (see **Step 0** below):
`python evaluation/run_exam.py <label> --write-bias-map` regenerates
`results.json`; the endpoint only reads that file. (Until Step 0 it ran
on 8 placeholder images and said so.)

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
> panel's own copy now discloses this directly, and since Step 0 it
> no longer names a tool at all: it answers "can't tell which tool made
> this", because on fresh images its confidence turned out not to
> separate its 8 tools from others (see **Step 0** below).

**Idea 3 — Live temporal (webcam) detection** was deliberately scoped
*out* of this phase — it needs real-time video infrastructure this
project doesn't have yet. Not started, not attempted.

### Step 0 — the exam set (measure first, then build)

The roadmap turns the three engines into one layered trust check
(Content Credentials → watermarks → two detectors → known-content
match → one sealed report). Before adding any layer, the project needed
a fair way to tell whether a layer *helps* — so Step 0 built the exam
first. Everything is in `deepguard-bouncer/evaluation/` (see its
README):

- **The exam set** — labelled images the detector was never tuned on:
  OpenFake's held-out test split (generators no training split
  contains, including GPT Image 1.5/2, Nano Banana Pro, Midjourney 7,
  Seedream 5, and frames from Sora 2 and Veo 3), OpenFake's Reddit set
  (pictures as they circulate), older GANs and diffusion models
  (CNNDetection, GenImage), our 44 outside images and 15 Commons
  pictures. Alongside: the C2PA public test files and watermarked
  samples for the layers still to come, and messaging-app-style copies.
  Every image's source is pinned in `exam_set_manifest.csv` with its
  licence and SHA-256; the images themselves live outside the repo.
- **A separate calibration set** — the only place anything is fitted.
- **`run_exam.py`** scores every exam image through `POST /api/analyze`
  and writes the report card; **`snapshot.py`** records the running
  app's answers and screenshots before and after every change.

**What today's system scores** is in `evaluation/BASELINE.md` — the
numbers every later step is judged against.

Three additive changes came with it: the source panel can now answer
**"can't tell which tool made this"** — and, because its confidence
turned out to be no guide to whether it knows the tool (AUROC below
0.5 on the calibration set), the rule written into
`models/detector_calibration.json` switches naming off entirely;
`/api/analyze` reports a **temperature-scaled
reading** next to the raw one (`calibrated_probability_fake`; the
verdict still uses the raw number until the trust report, Step 1,
re-derives its thresholds on calibrated numbers); and the **Bias Map**
now shows the exam set instead of 8 placeholder images.
`app/evidence.py` defines the one answer format every layer of the
trust check will return.

### Stage 3 — the provenance ledger

A detector can only guess. The ledger *proves* where a picture came
from — if its creator wrote it down first. It lives in
`deepguard-ledger/`, beside the bouncer, without a single bouncer file
changed; [its README](deepguard-ledger/README.md) has the details.

- **Registering** a file writes its three fingerprints (SHA-256, pHash,
  and the hash of its SSCD vector), its creator's claim (camera
  original, edited, or AI-generated) and what the detector saw, signed
  with the creator's own Ed25519 key. The picture itself is never stored.
- **Blocks.** Records are sealed into blocks: a Merkle root over the
  records plus the previous block's hash, signed by Deep-Guard. Change
  one letter in an old block and the checks fail from that block on.
- **Witnesses.** Two separate servers check every block, write its hash
  in their own notebooks and co-sign it. Even Deep-Guard, holding the
  real key, can't quietly rewrite history — the idea behind Certificate
  Transparency.
- **Four verdicts, provenance first:** 1 authentic & unaltered (the exact
  file); 2 the same picture in a modified file (re-saved or cropped);
  3 no provenance (only now does the detector run, labelled a
  prediction); 4 registered as AI-generated (honest synthetic content).
- **Check it yourself.** The Explorer recomputes every hash, Merkle root
  and signature in your own browser; `tools/verify_ledger.py` does it
  offline; a receipt proves one record with no server at all; a tamper
  lab runs five attacks on a copy of the chain.

### Stage 4 — durable provenance

A Stage 3 record holds while a copy still looks like the original and
everyone trusts the laptop. Stage 4 makes it hold up outside, in the
places pictures actually travel:

| Feature | The question it answers | How |
|---|---|---|
| **Anchored in Bitcoin** | "Could Deep-Guard and its witnesses agree to backdate a record?" | A block's hash is written into Bitcoin through OpenTimestamps. Your browser replays the proof and looks the Bitcoin block up itself. Block #2 of the demo ledger is in Bitcoin block 974545. |
| **A hidden ID in the pixels** | "WhatsApp stripped everything — whose photo is this?" | Registering hands back a *credentialed copy* carrying an invisible 61-bit TrustMark ID. Reading any later copy's pixels gives the ID back, and the ID points at the record — but only counts when the picture's content agrees. It survived a WhatsApp re-save, an 80% crop sent through WhatsApp, and a screenshot, on all 55 test images. |
| **Content Credentials (C2PA)** | "Can other apps read our label, and can we read theirs?" | Every checked file is read for a C2PA manifest against the official trust lists, and the credentialed copy carries one. Adobe's own Verify site reads ours: "This image was generated with an AI tool", for a picture declared AI-made. |
| **Deep-Guard Lens** | "Do I have to download a picture to check it?" | A Chrome/Edge extension: right-click any picture on any website, and a badge shows the verdict on the picture itself. |
| **A live public link** | "Can my audience try it on their own phones?" | `run_deepguard.py --public` opens a Cloudflare quick tunnel and puts a QR code on the projector. Visitors get fair-use limits and none of the owner's actions. |

The details: [the ledger README](deepguard-ledger/README.md) and
[the Lens README](deepguard-lens/README.md).

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
│   ├── attribution.py          # Idea 2 — generator attribution inference (+ "can't tell which tool")
│   ├── calibration.py          # Step 0 — does "90%" mean right 9 times in 10? (temperature scaling)
│   ├── evidence.py             # Step 0 — the one answer format every trust-check layer will return
│   ├── requirements.txt
│   └── static/                 # Plain HTML/CSS/JS frontend, no build step
├── bias_map/
│   ├── labeled_test_set.csv    # labels of the original 8 test images (now part of the exam set)
│   └── results.json            # Written by evaluation/run_exam.py --write-bias-map; read by /api/bias-map
├── evaluation/                 # Step 0 — the exam set, its scorer, before/after snapshots (see its README)
│   ├── build_exam_set.py       # Builds/checks the exam + calibration sets (images live outside the repo)
│   ├── run_exam.py             # Scores the exam through /api/analyze; writes the report card
│   ├── snapshot.py             # Records the running app's answers + screenshots; compares two recordings
│   ├── exam_set_manifest.csv   # Every exam image: source (pinned), label, licence, SHA-256
│   ├── calibration_set_manifest.csv
│   ├── BASELINE.md             # Today's system on the exam — the numbers every later step is judged against
│   └── results/                # One folder per scored run
├── colab_notebook/
│   ├── DeepGuard_Training_MASSIVE.ipynb        # Trains deepguard_bouncer.pth
│   └── DeepGuard_Attribution_Training.ipynb    # Trains deepguard_attribution.pth
├── demo_artworks/              # Seed images for the Content Registry
├── tools/
│   ├── build_concept_labeler.py   # One-time build of the registry's CLIP namer (optional)
│   └── verify_offline.py          # Check a signed record with only a public key, no server
└── models/                     # See PUT_WEIGHTS_HERE.md — the three core files included,
                                #   plus detector_calibration.json (fitted by run_exam.py --fit)

deepguard-ledger/               # Stages 3 and 4, beside the bouncer (whose app it imports unchanged)
├── run_deepguard.py            # Starts the server and two witnesses (--public: also the public link)
├── ledger_app.py               # Adds /ledger, /explorer, /share and /api/ledger/* to the bouncer's app
├── ledger_core.py              # Blocks, hash chain, Merkle proofs, signatures, verification, tamper lab
├── creators.py                 # One Ed25519 key pair per creator (a simulated PKI)
├── witness.py                  # An independent witness: checks, records and co-signs every block
├── anchoring.py                # Stage 4 — Bitcoin anchors through OpenTimestamps
├── watermark.py                # Stage 4 — the hidden ID in the pixels (TrustMark)
├── credentials.py              # Stage 4 — Content Credentials (C2PA): read, verify, sign
├── public_mode.py, tunnel.py   # Stage 4 — the public link: visitors' rules, Cloudflare tunnels
├── static/                     # The pages; dgcrypto.js redoes every check in the browser
├── tools/                      # verify_ledger.py (offline checks), measure_watermark.py, update_trust_lists.py
├── trust/                      # The C2PA trust lists, pinned
├── measurements/               # watermark_robustness.json — the hidden ID on 55 images
└── tests/                      # 112 tests

deepguard-lens/                 # Stage 4 — the browser extension (load it unpacked in Edge or Chrome)
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

119 tests: unit tests for the pure-logic modules (fingerprint hashing/
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
explanations existed, a stale features file). The offline verifier
(`tools/verify_offline.py`) is checked against the server's own signing
code and against real server answers saved to disk, including a forger
who embeds their own key in a record. Step 0's pieces are pinned too:
temperature scaling (it never moves an image across 50%, and the fit
recovers a known temperature), the answer format's rules (e.g. only a
verified signature may count as proof), and the source panel's
"unknown" answer. The evaluation tools have their own 17 tests
(`python -m pytest evaluation/test_evaluation_tools.py` from
`deepguard-bouncer/`) — the scorer's arithmetic, the source panel's
naming rule, the file-name collision guard, WhatsApp copies being paired
with their originals, and the before/after tool catching a changed value
(not just a changed field). The Content Registry's
persisted files are redirected to a temp directory for the test run —
running the suite never touches the real `deepguard-bouncer/models/content_registry*`
files. Takes about 40 seconds, dominated by loading the real model
weights once at session start.

The ledger has its own 112 tests, from `deepguard-ledger/`:

```bash
python -m pytest tests -q
```

They cover the ledger's maths and every tamper-lab attack, Bitcoin anchors
(against the official OpenTimestamps example and the demo ledger's real
proof), Content Credentials (against the official C2PA test files), the
hidden ID (a WhatsApp copy, a crop, an ID lifted onto another picture), the
public link's rules, and the whole app with real models and real witness
servers. They use a throwaway folder, so they never touch your ledger.

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

- **The detector misses generators it never saw.** On the exam set
  (`evaluation/BASELINE.md`) it calls 58% of AI images AI-generated,
  27% *authentic*, and false-alarms on 14% of real photos (AUC 0.76):
  34 of 34 DALL-E 3 images, 6 of 10 each for Midjourney 7, GPT Image 2
  and Nano Banana Pro, and 25% on older research generators. The file's
  size makes little difference (the detector resizes everything to
  256×256 first); what it cannot handle is generators — and their
  formats and subjects — that it never saw. The fix is training data
  from many generators — roadmap Step 4.
- **The detector's percentages overclaim.** Of images it scores below
  10% AI, 39% really are AI; honest (temperature-scaled, T = 10.6)
  readings are almost always between 20% and 80%. The page still shows
  the raw verdicts until the trust report (Step 1) re-words them on
  calibrated numbers; `/api/analyze` already reports both.
- **The source panel does not identify tools on fresh images.** On
  fresh images of its own 8 tools it named the right one only 39% of
  the time, and it is *more* confident on tools it has never seen
  (AUROC 0.40 — see `generalization_test/` for the first sign of this).
  So it no longer names a tool: it answers "can't tell which tool made
  this" for every image. A useful answer needs a retrained source model.
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
- **The ledger's identities are simulated.** The server holds every
  creator's private key, like a custodial wallet, and nobody checks who a
  creator really is. The ledger proves who wrote a picture down *first*,
  not who took it — that is what disputes are for.
- **The witnesses run on the same laptop.** They are separate processes
  with separate keys and notebooks, enough to show the mechanism; real
  witnesses would be other people's machines (the next planned step).
- **Our Content Credentials come from a demo certificate authority**, so
  other C2PA tools show our manifest but call its issuer "unrecognized":
  only members of the C2PA conformance programme are on the official list.
- **The hidden ID can be removed** by TrustMark's own public remover model;
  that is why it only ever points *to* a record whose fingerprints must
  also match.
- **The public link is a Cloudflare quick tunnel**: free and account-less,
  but with no uptime guarantee and a new address on every start. The Lens
  extension is loaded by hand, not from the Chrome or Edge stores.
