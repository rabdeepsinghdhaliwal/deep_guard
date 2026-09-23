"""
Deep-Guard — local media analysis server.

Run with (from the app/ directory, inside your activated venv):
    uvicorn main:app --reload

Then open http://127.0.0.1:8000 in a browser.

What this file does, in order:
  1. On startup, loads the fine-tuned .pth checkpoint produced by the
     Colab notebook and builds the matching model architecture.
  2. Serves the static frontend (static/index.html, style.css, script.js)
     at "/".
  3. Exposes POST /api/analyze — accepts an uploaded image or video.
     Images are scored directly. Videos are sampled across their full
     duration (up to MAX_FRAMES_SAMPLED frames), every sampled frame is
     scored, and the results are aggregated into a single verdict plus a
     per-frame timeline showing where in the clip anything was detected.
  4. Exposes GET /api/health — a diagnostic endpoint the frontend pings
     on load, so a broken setup is visible immediately instead of
     failing silently on first upload.
  5. Every /api/analyze response also carries an authenticity fingerprint
     (see fingerprint.py — Stage 2): an exact SHA-256 hash, a perceptual
     hash that survives re-compression, and both signed with this
     server's Ed25519 key. This is independent of the AI verdict above —
     it doesn't check anything against a registry (that's Stage 3, not
     built yet), it just produces and signs the fingerprint an upload
     WOULD be registered under, so the mechanism is demoable today.
  6. Exposes POST /api/fingerprint/compression-test — re-saves an
     uploaded image at several JPEG qualities server-side and reports
     both fingerprints for each, live, so the "exact hash breaks, visual
     hash survives" claim is a one-click browser feature, not something
     that only shows up when someone runs a script.
  7. Every image explainability result also carries a frequency-domain
     panel (see frequency_analysis.py — Phase 2, Idea 5b): a Fourier
     spectrum of the exact image the model saw, plus any anomalous
     frequency peaks detected in it. This needs no model at all — real
     cameras and AI generators leave measurably different traces in an
     image's frequency spectrum, so this is independent physical
     evidence shown alongside Grad-CAM's model-derived heatmap, not a
     replacement for it.
  8. Exposes POST /api/explain/shap-regions — an opt-in, separate-click
     breakdown (see shap_explain.py — Phase 2, Idea 5a) of exactly how
     many percentage points each region of an image pushed the verdict
     toward fake or real, using Shapley values over superpixel regions.
     Kept as its own endpoint rather than folded into /api/analyze
     because it costs several seconds of extra CPU inference — worth it
     for a deliberate "show me why" click, not for every upload.
  9. Exposes GET /bias-map and GET /api/bias-map (Phase 2, Idea 1): a
     dashboard showing accuracy sliced by generator/subject/style,
     instead of one aggregate number that can hide a subgroup the model
     does badly on (exactly how this project's own hidden 91.5%/8.5%
     split was found). This endpoint only READS bias_map/results.json —
     see bias_map/build_bias_map.py, a standalone script run by hand,
     for how that file gets produced.
 10. Exposes GET /registry, POST /api/registry/register, and POST
     /api/registry/check (Phase 2, Idea 4 — see content_registry.py):
     the first real piece of Stage 3 (the ledger), upgraded from
     "was this exact file registered" to "was any registered content
     used as the basis for this file, even cropped, rotated, or
     recoloured" — using Meta's pretrained SSCD model plus a FAISS
     index, not the perceptual hash above (which is a global hash of
     the whole frame and breaks under cropping almost immediately).
 11. Every image /api/analyze verdicts as "manipulated" also carries a
     generator_attribution field (Phase 2, Idea 2 — see attribution.py
     and model.py's GENERATOR_CLASSES): which of 8 specific tools
     (4 GAN, 4 diffusion) most likely produced it, from a SEPARATE
     model trained on ArtiFact (not Stage 1's detector, which never
     changes). Requires models/deepguard_attribution.pth to exist —
     additive, like everything else here; absent that checkpoint this
     field is simply never present, not a broken app.
 12. The registry explains itself (see registry_explain.py): every
     registered or checked image gets a distinct_features summary
     (distinctive points, named 3x3 areas, colours), and every match an
     explanation — which named parts the two images share, how many
     distinctive points confirm it, what edit was made, and how much of
     the SSCD score each shared part is responsible for. Names come from
     an optional CLIP "namer" (models/concept_labeler_*, built by
     tools/build_concept_labeler.py); without it areas are named by
     position. Explanations never change which images match.
"""

import io
import json
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

import base64

import attribution
import concept_labels
import content_registry
import fingerprint
import frequency_analysis
import gradcam
import local_features
import registry_explain
import shap_explain
import uncertainty
from model import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepguard")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_bouncer.pth"
SIGNER_PRIVATE_KEY_PATH = PROJECT_ROOT / "models" / "demo_signer_private.pem"
SIGNER_PUBLIC_KEY_PATH = PROJECT_ROOT / "models" / "demo_signer_public.pem"
BIAS_MAP_RESULTS_PATH = PROJECT_ROOT / "bias_map" / "results.json"
SSCD_MODEL_PATH = PROJECT_ROOT / "models" / "sscd_disc_mixup.torchscript.pt"
CONTENT_REGISTRY_INDEX_PATH = PROJECT_ROOT / "models" / "content_registry.index"
CONTENT_REGISTRY_METADATA_PATH = PROJECT_ROOT / "models" / "content_registry_metadata.json"
CONTENT_REGISTRY_FEATURES_DIR = PROJECT_ROOT / "models" / "content_registry_features"
DEMO_ARTWORKS_DIR = PROJECT_ROOT / "demo_artworks"
ATTRIBUTION_MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_attribution.pth"
CONCEPT_LABELER_ENCODER_PATH = PROJECT_ROOT / "models" / "concept_labeler_image_encoder.torchscript.pt"
CONCEPT_LABELER_VOCAB_PATH = PROJECT_ROOT / "models" / "concept_labeler_vocabulary.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB — videos are now analysed in full

# How many frames we sample across a clip. 32 is a deliberate balance:
# enough temporal coverage to catch a manipulation that appears in only
# part of the video, few enough that a CPU-only machine returns a result
# in seconds rather than minutes.
MAX_FRAMES_SAMPLED = 32

# Frames are pushed through the model in batches rather than one at a
# time — far fewer forward passes for the same work.
INFERENCE_BATCH_SIZE = 8

# Populated at startup by load_model(). Kept as simple module state
# rather than a class — this is a single-model, single-purpose service.
STATE = {
    "model": None,
    "device": None,
    "class_order": None,
    "input_size": None,
    "normalize_mean": None,
    "normalize_std": None,
    "preprocess": None,
    "load_error": None,
    "signer_private_key": None,
    "signer_public_key": None,
    "sscd_model": None,
    "content_registry": None,
    "attribution": None,
    "concept_labeler": None,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Loads the checkpoint saved by the Colab notebook and prepares the
    model for inference. Any failure here is caught and stored in
    STATE["load_error"] rather than raised, so the server still starts
    and /api/health can report exactly what's wrong instead of the
    process crashing with a traceback the user has to decode.
    """
    if not MODEL_PATH.exists():
        STATE["load_error"] = (
            f"No model file found at {MODEL_PATH}. It is part of the "
            f"repository, so restore it (git checkout -- "
            f"deepguard-bouncer/models/deepguard_bouncer.pth) or clone "
            f"again; to retrain it instead, see the README's Retraining "
            f"section and models/PUT_WEIGHTS_HERE.md."
        )
        logger.error(STATE["load_error"])
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        if "model_state_dict" not in checkpoint:
            # The training notebook also writes a raw state_dict (no
            # metadata) to the same filename mid-run, before the final
            # cell overwrites it with this wrapped format. If someone
            # downloads that intermediate file by mistake, treat the
            # whole checkpoint as the state_dict itself rather than
            # failing on a confusing KeyError below.
            checkpoint = {"model_state_dict": checkpoint}

        architecture = checkpoint.get("architecture", "efficientnet_b0")
        if architecture != "efficientnet_b0":
            raise ValueError(
                f"Checkpoint declares architecture '{architecture}', but "
                f"this server only knows how to build 'efficientnet_b0'. "
                f"Did the wrong .pth file get copied in?"
            )

        model = build_model(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        class_order = checkpoint.get("class_order", ["real", "fake"])
        input_size = checkpoint.get("input_size", 224)
        mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
        std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

        # IMPORTANT: this must match the training notebook's eval_transforms
        # exactly. The notebook resizes to a SQUARE 256x256 (which distorts
        # aspect ratio) before centre-cropping. If we resized by the shorter
        # side here instead, the model would see images shaped differently at
        # inference than it did during training, and accuracy would quietly
        # drop for no visible reason.
        resize_to = int(input_size * 256 / 224)
        preprocess = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        STATE.update({
            "model": model,
            "device": device,
            "class_order": class_order,
            "input_size": input_size,
            "normalize_mean": mean,
            "normalize_std": std,
            "preprocess": preprocess,
            "load_error": None,
        })
        logger.info(f"Model loaded from {MODEL_PATH} onto device={device}")

    except Exception as exc:  # noqa: BLE001 — deliberately broad; see docstring
        STATE["load_error"] = f"Failed to load model: {exc!r}"
        logger.exception("Model load failed")


def load_or_create_signing_key() -> None:
    """
    Loads the demo signing keypair from disk, generating and persisting one
    on first run. Persisting it (rather than generating a fresh key every
    startup) means the same "signer" identity is used across restarts —
    otherwise a fingerprint signed before a restart would appear to come
    from a different signer than one signed after, which would break the
    "same server, same signer" story before Stage 3 (the registry) even
    exists to make that story matter.
    """
    try:
        if SIGNER_PRIVATE_KEY_PATH.exists() and SIGNER_PUBLIC_KEY_PATH.exists():
            private_pem = SIGNER_PRIVATE_KEY_PATH.read_bytes()
            public_pem = SIGNER_PUBLIC_KEY_PATH.read_bytes()
        else:
            private_pem, public_pem = fingerprint.generate_keypair()
            SIGNER_PRIVATE_KEY_PATH.write_bytes(private_pem)
            SIGNER_PUBLIC_KEY_PATH.write_bytes(public_pem)
            logger.info(f"Generated new demo signing key at {SIGNER_PRIVATE_KEY_PATH}")

        STATE["signer_private_key"] = private_pem
        STATE["signer_public_key"] = public_pem
    except Exception:  # noqa: BLE001 — fingerprinting is additive; don't crash startup over it
        logger.exception("Failed to load or create signing key — fingerprints will not be signed")


def load_content_registry() -> None:
    """
    Loads the pretrained SSCD model and the persisted FAISS registry
    (Phase 2, Idea 4). On first run only, seeds the registry with a few
    public-domain artworks (demo_artworks/) so the "register → crop it →
    still gets caught" demo has something in it immediately, rather than
    starting empty. Seeding is idempotent via is_registered() — it will
    not re-add the same title on a later restart.

    Entirely additive, like signing: if the SSCD weights are missing or
    fail to load, the registry endpoints report 503 rather than this
    taking down the AI-detection half of the app, which doesn't depend
    on it at all.
    """
    if not SSCD_MODEL_PATH.exists():
        logger.warning(f"No SSCD model at {SSCD_MODEL_PATH} — content registry disabled.")
        return

    try:
        STATE["sscd_model"] = content_registry.load_sscd_model(SSCD_MODEL_PATH)
        registry = content_registry.ContentRegistry(
            CONTENT_REGISTRY_INDEX_PATH, CONTENT_REGISTRY_METADATA_PATH, features_dir=CONTENT_REGISTRY_FEATURES_DIR,
        )
        STATE["content_registry"] = registry

        if DEMO_ARTWORKS_DIR.exists():
            for image_path in sorted(DEMO_ARTWORKS_DIR.glob("*.jpg")):
                title = image_path.stem.replace("_", " ").title()
                if registry.is_registered(title):
                    _backfill_demo_features(registry, title, image_path)
                    continue
                image = Image.open(image_path).convert("RGB")
                embedding = content_registry.compute_content_embedding(STATE["sscd_model"], image)
                file_bytes = image_path.read_bytes()
                record = fingerprint.build_and_sign_record(
                    STATE["signer_private_key"], STATE["signer_public_key"],
                    **fingerprint.compute_file_fingerprint(file_bytes),
                    title=title,
                    source="Wikimedia Commons (public domain)",
                )
                registry.register(embedding, record, _registry_features(image))
                logger.info(f"Seeded content registry with '{title}'")

        logger.info(f"Content registry loaded: {len(registry.metadata)} entries")
    except Exception:  # noqa: BLE001 — additive; don't crash startup over it
        logger.exception("Failed to load content registry — /api/registry/* will report unavailable")


def _registry_features(image: Image.Image) -> Optional[dict]:
    """What a registration stores for explaining future matches (see
    registry_explain.storable), or None if describing the image failed —
    the registration itself must never fail over its explanation."""
    try:
        upload = registry_explain.describe(image, STATE["concept_labeler"], local_features.MAX_STORED_POINTS)
        return registry_explain.storable(upload, STATE["concept_labeler"])
    except Exception:  # noqa: BLE001 — additive
        logger.exception("Couldn't extract distinct features — registering without an explanation")
        return None


def _backfill_demo_features(registry: content_registry.ContentRegistry, title: str, image_path: Path) -> None:
    """The demo artworks were first registered before the registry kept
    distinctive points. Their pixels are right here in demo_artworks/, so
    their features can be rebuilt — and rebuilt again whenever the namer's
    vocabulary or the feature format changes, so their area names never go
    stale. (A user's own earlier registration can't be backfilled: the
    registry never kept their image.)"""
    labeler = STATE["concept_labeler"]
    current = (labeler.vocab_hash if labeler is not None else None, registry_explain.FEATURE_VERSION)
    for entry_id in registry.entry_ids_titled(title):
        stored = registry.load_features(entry_id)
        if stored is not None:
            meta = registry_explain.stored_meta(stored)
            if (meta.get("vocab_hash"), meta.get("feature_version")) == current:
                continue
        features = _registry_features(Image.open(image_path).convert("RGB"))
        if features is not None:
            registry.save_features(entry_id, features)
            logger.info(f"Rebuilt distinct features for registry entry {entry_id} ('{title}')")


def load_concept_labeler() -> None:
    """
    Loads the registry's optional "namer" (see concept_labels.py): CLIP's
    image half plus a pre-encoded vocabulary, built once by
    tools/build_concept_labeler.py. Additive, like every model here:
    without it, registry explanations name areas by position ("top left
    area") instead of by content, and nothing else changes.
    """
    try:
        labeler = concept_labels.load_concept_labeler(CONCEPT_LABELER_ENCODER_PATH, CONCEPT_LABELER_VOCAB_PATH)
        if labeler is None:
            logger.info("No concept labeler in models/ — registry areas will be named by position. "
                        "Run tools/build_concept_labeler.py once to enable names.")
            return
        STATE["concept_labeler"] = labeler
        logger.info(f"Concept labeler loaded: {labeler.model_card}, {len(labeler.thing_labels)} names")
    except Exception:  # noqa: BLE001 — additive; don't crash startup over it
        logger.exception("Failed to load the concept labeler — registry areas will be named by position")


def load_attribution_model() -> None:
    """
    Loads the Generator Attribution checkpoint (Phase 2, Idea 2) if one
    has been trained and placed at ATTRIBUTION_MODEL_PATH. Additive, like
    every other optional capability here: no checkpoint means /api/analyze
    simply never adds a generator_attribution field, not a broken app —
    this is expected right up until the Colab training notebook has
    actually been run once.
    """
    try:
        state = attribution.load_attribution_model(ATTRIBUTION_MODEL_PATH, STATE["device"] or torch.device("cpu"))
        if state is None:
            logger.info(f"No attribution checkpoint at {ATTRIBUTION_MODEL_PATH} — generator attribution disabled.")
            return
        STATE["attribution"] = state
        logger.info(f"Attribution model loaded: {len(state['class_order'])} generator classes")
    except Exception:  # noqa: BLE001 — additive; don't crash startup over it
        logger.exception("Failed to load attribution model — generator_attribution will be unavailable")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_or_create_signing_key()
    load_concept_labeler()   # before the registry, so seeded artworks get named areas
    load_content_registry()
    load_attribution_model()
    yield


app = FastAPI(title="Deep-Guard", lifespan=lifespan)

# Permissive CORS for local development only. If you later split the
# frontend onto a different origin (e.g. a live-server on :5500), this
# is why cross-origin calls to :8000 will still work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/bias-map")
async def serve_bias_map_page():
    return FileResponse(str(STATIC_DIR / "bias-map.html"))


@app.get("/registry")
async def serve_registry_page():
    return FileResponse(str(STATIC_DIR / "registry.html"))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok" if STATE["model"] is not None else "model_not_loaded",
        "model_loaded": STATE["model"] is not None,
        "device": str(STATE["device"]) if STATE["device"] else None,
        "error": STATE["load_error"],
        "signer_public_key": (
            STATE["signer_public_key"].decode("ascii") if STATE["signer_public_key"] else None
        ),
    }


@app.get("/api/bias-map")
async def bias_map():
    """
    Reads bias_map/results.json and returns it as-is — nothing is
    computed on the request path. That file is written by hand-running
    bias_map/build_bias_map.py, not by this endpoint, so this stays
    fast and safe to call regardless of how large the labeled test set
    ever grows. Returns a clear "not generated yet" response rather
    than a 404 or a crash if the script has never been run.
    """
    if not BIAS_MAP_RESULTS_PATH.exists():
        return {
            "generated": False,
            "message": "No bias map has been generated yet. Run bias_map/build_bias_map.py first.",
        }
    return {"generated": True, **json.loads(BIAS_MAP_RESULTS_PATH.read_text(encoding="utf-8"))}


# ---------------------------------------------------------------------------
# Content registry (Phase 2, Idea 4)
# ---------------------------------------------------------------------------

def _check_upload_size(file_bytes: bytes) -> None:
    """
    Shared by every upload endpoint. Found during the round-1 adversarial
    review: /api/analyze was the ONLY endpoint enforcing MAX_UPLOAD_BYTES —
    compression-test, shap-regions, and both registry endpoints all read
    the full upload into memory with no size check at all, an unbounded
    memory-exhaustion gap on four of five upload paths. Centralised here
    so a sixth upload endpoint can't reintroduce the same gap by omission.
    """
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) / 1e6:.1f} MB). Limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )


MAX_TITLE_LENGTH = 200


def _clean_registry_title(title: str) -> str:
    """
    Found during the round-1 review: titles were stored completely
    unvalidated — empty, whitespace-only, or arbitrarily long strings were
    all accepted as-is. Trimmed and length-capped here, at the one place
    a title enters the system, rather than trusting every future caller
    to remember to check.
    """
    cleaned = title.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Title can't be empty.")
    if len(cleaned) > MAX_TITLE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Title is too long (limit {MAX_TITLE_LENGTH} characters).")
    return cleaned


def _read_registry_upload_image(file_bytes: bytes) -> Image.Image:
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    _check_upload_size(file_bytes)
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Couldn't read that image: {exc}") from exc


@app.post("/api/registry/register")
async def registry_register(file: UploadFile = File(...), title: str = Form(...)):
    """
    Registers an image's content in the registry: computes its SSCD
    embedding (for future crop/edit-robust lookups) AND its existing
    SHA-256 + perceptual hash fingerprint, signs the combined record
    with this server's Ed25519 key (reusing fingerprint.py exactly as
    /api/analyze does), and adds the embedding to the FAISS index.
    Also keeps the image's distinct features (distinctive points, named
    areas, colour grid — no pixels) so later matches can be explained,
    and returns a summary of them as `distinct_features`.
    """
    if STATE["sscd_model"] is None or STATE["content_registry"] is None:
        raise HTTPException(status_code=503, detail="Content registry is unavailable on this server.")

    title = _clean_registry_title(title)
    file_bytes = await file.read()
    image = _read_registry_upload_image(file_bytes)

    embedding = content_registry.compute_content_embedding(STATE["sscd_model"], image)
    record = fingerprint.build_and_sign_record(
        STATE["signer_private_key"], STATE["signer_public_key"],
        **fingerprint.compute_file_fingerprint(file_bytes),
        phash=fingerprint.compute_image_phash(image),
        title=title,
        source="user upload",
    )
    upload = _describe_upload(image, local_features.MAX_STORED_POINTS)
    features = None
    if upload is not None:
        try:
            features = registry_explain.storable(upload, STATE["concept_labeler"])
        except Exception:  # noqa: BLE001 — additive: register without an explanation rather than not at all
            logger.exception("Couldn't prepare distinct features for storage")
            upload = None
    STATE["content_registry"].register(embedding, record, features)

    # The signed record is returned exactly as stored; distinct_features is
    # added alongside it (the same pattern as `similarity` on a match), so
    # verifying the record means verifying everything except that key.
    response = dict(record)
    if upload is not None:
        response["distinct_features"] = {
            **upload.summary,
            "stored_kb": registry_explain.stored_size_kb(features),
        }
    return response


@app.post("/api/registry/check")
async def registry_check(file: UploadFile = File(...)):
    """
    Searches the registry for anything registered that shares
    substantial content with the uploaded image — including a crop,
    rotation, recolour, or heavy recompression of it. Returns matches
    (empty list if none clear MATCH_THRESHOLD) with their similarity
    score and the original signed registration record, plus for each an
    `explanation` (what the two share, what edit was made, how much of
    the score each shared part is responsible for), and the upload's own
    `upload_features`. Which images match is decided by SSCD alone.
    """
    if STATE["sscd_model"] is None or STATE["content_registry"] is None:
        raise HTTPException(status_code=503, detail="Content registry is unavailable on this server.")

    file_bytes = await file.read()
    image = _read_registry_upload_image(file_bytes)

    registry = STATE["content_registry"]
    embedding, feature_map = content_registry.compute_content_features(STATE["sscd_model"], image)
    matches = registry.search(embedding)
    upload = _describe_upload(image, local_features.MAX_QUERY_POINTS)
    for match in matches:
        match["explanation"] = _explain_match(registry, upload, match, feature_map)

    response = {"matches": matches, "registry_size": registry.index.ntotal}
    if upload is not None:
        response["upload_features"] = upload.summary
    return response


def _describe_upload(image: Image.Image, max_points: int) -> Optional["registry_explain.Upload"]:
    try:
        return registry_explain.describe(image, STATE["concept_labeler"], max_points)
    except Exception:  # noqa: BLE001 — additive: a registry answer must never fail over its explanation
        logger.exception("Couldn't describe the upload's distinct features")
        return None


def _explain_match(registry: content_registry.ContentRegistry, upload, match: dict, feature_map) -> dict:
    if upload is None:
        return {"available": False, "reason": "The upload's distinctive points couldn't be extracted."}
    try:
        entry_id = match["entry_id"]
        return registry_explain.explain_match(
            upload, match["similarity"], registry.load_features(entry_id), STATE["sscd_model"],
            registry.stored_vector(entry_id), feature_map,
        )
    except Exception:  # noqa: BLE001 — additive
        logger.exception("Couldn't explain a registry match")
        return {"available": False, "reason": "Something went wrong explaining this match; the match itself stands."}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def sample_frames_from_video(
    video_bytes: bytes, extension: str, max_frames: int = MAX_FRAMES_SAMPLED
) -> tuple[list[Image.Image], list[float], dict]:
    """
    Samples frames spread evenly across the whole clip and returns
    (frames, timestamps_in_seconds, video_metadata).

    Why sample by POSITION rather than take the first N frames: a
    deepfake is rarely uniformly bad across a clip. Face-swap artifacts
    spike when the subject turns, blinks, or is partly occluded, and
    stay invisible in easy front-facing moments. Sampling across the
    full duration is what lets us catch those spikes — and it's why
    the per-frame timeline we return is more informative than any
    single aggregate number.

    We skip the first and last 5% of the clip: fades, slates and black
    frames there carry no face and would drag the average toward noise.

    The temp file keeps the SAME extension as the upload because some
    OpenCV builds select their demuxer by file extension — saving a
    .webm as "*.mp4" can silently produce a file OpenCV refuses to open.
    """
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError(
                "Could not open that video — it may be corrupt or use an unsupported codec."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / fps) if fps > 0 else 0.0

        if total_frames <= 0:
            # Some containers don't report a frame count. Fall back to
            # reading sequentially rather than failing outright.
            frames, timestamps = [], []
            index = 0
            while len(frames) < max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
                timestamps.append(index / fps if fps > 0 else float(index))
                index += 1
            cap.release()
            if not frames:
                raise ValueError("No readable frames were found in that video.")
            return frames, timestamps, {
                "total_frames": len(frames), "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
            }

        start = int(total_frames * 0.05)
        end = int(total_frames * 0.95)
        if end <= start:
            start, end = 0, max(total_frames - 1, 0)

        n = min(max_frames, max(end - start, 1))
        indices = [start + round(i * (end - start) / max(n - 1, 1)) for i in range(n)] if n > 1 else [start]

        frames, timestamps = [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            timestamps.append(round(idx / fps, 2) if fps > 0 else float(idx))

        cap.release()

        if not frames:
            raise ValueError("No readable frames could be decoded from that video.")

        return frames, timestamps, {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def score_frames(frames: list[Image.Image]) -> list[float]:
    """
    Runs the model over a list of frames in batches and returns P(fake)
    for each. Batching matters: on CPU, 32 separate forward passes cost
    noticeably more than a handful of batched ones.
    """
    probabilities: list[float] = []
    preprocess = STATE["preprocess"]
    device = STATE["device"]
    model = STATE["model"]

    with torch.no_grad():
        for i in range(0, len(frames), INFERENCE_BATCH_SIZE):
            chunk = frames[i : i + INFERENCE_BATCH_SIZE]
            batch = torch.stack([preprocess(f) for f in chunk]).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            if isinstance(probs, float):  # a batch of exactly 1 squeezes to a scalar
                probs = [probs]
            probabilities.extend(probs)

    return probabilities


def aggregate_video_scores(probabilities: list[float]) -> dict:
    """
    Turns per-frame probabilities into one verdict.

    We report BOTH the mean and the fraction of frames above 0.5,
    because they answer different questions. The mean asks "how fake
    does this clip look overall"; the flagged fraction asks "how much
    of the clip looks fake". A clip where 20% of frames are strongly
    manipulated and 80% are clean is a real and important case — a
    localised face swap — and a bare mean would bury it.

    The headline number is a blend weighted toward the mean, with the
    peak pulled in so a short but unmistakable burst of manipulation
    isn't averaged into invisibility.
    """
    n = len(probabilities)
    mean_p = sum(probabilities) / n
    peak_p = max(probabilities)
    flagged = sum(1 for p in probabilities if p >= 0.5)
    flagged_ratio = flagged / n

    ordered = sorted(probabilities)
    median_p = (
        ordered[n // 2] if n % 2 == 1
        else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    )

    # 75% mean + 25% peak. Documented deliberately so the threshold is
    # defensible rather than arbitrary.
    headline = 0.75 * mean_p + 0.25 * peak_p

    return {
        "probability_fake": round(headline, 4),
        "mean_probability": round(mean_p, 4),
        "median_probability": round(median_p, 4),
        "peak_probability": round(peak_p, 4),
        "min_probability": round(min(probabilities), 4),
        "frames_analysed": n,
        "frames_flagged": flagged,
        "flagged_ratio": round(flagged_ratio, 4),
    }


def build_fingerprint(file_bytes: bytes, phash_field: dict) -> Optional[dict]:
    """
    Computes and signs the authenticity fingerprint for an upload (Stage 2).
    Returns None (rather than raising) if the signing key never loaded —
    this is additive to the AI verdict, so a fingerprinting problem
    shouldn't take down /api/analyze's core function.
    """
    private_key = STATE["signer_private_key"]
    public_key = STATE["signer_public_key"]
    if private_key is None or public_key is None:
        logger.warning("Signing key unavailable — returning fingerprint unsigned fields only")
        return None

    try:
        file_fp = fingerprint.compute_file_fingerprint(file_bytes)
        record = fingerprint.build_and_sign_record(
            private_key, public_key, **file_fp, **phash_field
        )
        return record
    except Exception:  # noqa: BLE001 — fingerprinting is additive; never break analysis over it
        logger.exception("Fingerprinting failed")
        return None


# How many stochastic forward passes to average for the consistency check.
# 16 keeps the extra latency small (one batched forward call) while still
# giving a stable standard deviation estimate.
MC_DROPOUT_SAMPLES = 16


def build_explainability(image: Image.Image) -> Optional[dict]:
    """
    Computes the Grad-CAM heatmap and Monte Carlo Dropout consistency for a
    single image (for video, the caller passes the single most-suspicious
    sampled frame rather than running this per frame). Additive, like
    build_fingerprint — returns None on any failure rather than breaking
    the core AI verdict.
    """
    try:
        preprocess = STATE["preprocess"]
        model = STATE["model"]
        device = STATE["device"]

        tensor = preprocess(image).unsqueeze(0).to(device)

        heatmap = gradcam.compute_gradcam(model, tensor)
        # The overlay is drawn on exactly what the model saw (resized and
        # centre-cropped, before normalization) rather than the original
        # photo — see gradcam.py's docstring for why that's the honest
        # choice here. The frequency panel below reuses this same
        # model-view image for the same reason.
        model_view_transform = transforms.Compose(preprocess.transforms[:2])
        model_view_image = model_view_transform(image)
        heatmap_png = gradcam.render_heatmap_overlay(model_view_image, heatmap)

        mean_p, std_p = uncertainty.score_with_uncertainty(model, tensor, n_samples=MC_DROPOUT_SAMPLES)

        # Frequency-domain panel (Phase 2, Idea 5b) — pure signal
        # processing on the pixels, no model involved. Wrapped in its
        # own try/except: it's additive evidence alongside Grad-CAM, not
        # a dependency of it, so a failure here shouldn't blank out the
        # heatmap and consistency numbers above, which just succeeded.
        frequency_result = None
        try:
            spectrum = frequency_analysis.compute_fft_spectrum(model_view_image)
            peak_info = frequency_analysis.detect_spectral_peaks(spectrum, model_view_image)
            spectrum_png = frequency_analysis.render_spectrum_png(spectrum)
            frequency_result = {
                "spectrum_png_base64": base64.b64encode(spectrum_png).decode("ascii"),
                **peak_info,
            }
        except Exception:  # noqa: BLE001 — additive only
            logger.exception("Frequency-domain analysis failed")

        return {
            "heatmap_png_base64": base64.b64encode(heatmap_png).decode("ascii"),
            "frequency_analysis": frequency_result,
            "model_consistency": {
                "mean_probability": round(mean_p, 4),
                "std_probability": round(std_p, 4),
                # Empirically (see the retrain investigation), a confidently
                # WRONG verdict tends to show LOW std here, not high, because
                # the sigmoid is saturated near 0 or 1. A low std means the
                # model is internally consistent, not that it's correct.
                "note": "Low std means the model is internally consistent about this "
                        "reading -- not that the reading is correct. High std means "
                        "the model itself is unstable here; treat the headline number "
                        "with extra caution.",
            },
        }
    except Exception:  # noqa: BLE001 — additive only, never break the core verdict
        logger.exception("Explainability computation failed")
        return None


def build_generator_attribution(image: Image.Image) -> Optional[dict]:
    """
    Phase 2, Idea 2. Only meaningful to call when the Stage 1 verdict is
    already "manipulated" — the caller is responsible for that check, not
    this function, so this stays a pure "is the capability available"
    gate. Additive like everything else here: returns None if no
    checkpoint was ever loaded, never raises into the core verdict.
    """
    if STATE["attribution"] is None:
        return None
    try:
        return attribution.predict_generator(STATE["attribution"], image, STATE["device"])
    except Exception:  # noqa: BLE001 — additive only
        logger.exception("Generator attribution failed")
        return None


# JPEG quality levels used for the live compression-resilience demo below.
# 95 down to 30 spans "barely touched" to "heavily squeezed, WhatsApp-style."
COMPRESSION_TEST_QUALITIES = [95, 85, 70, 50, 30]


@app.post("/api/fingerprint/compression-test")
async def compression_test(file: UploadFile = File(...)):
    """
    Demonstrates the core Stage 2 claim live, in the browser, without
    needing an external script: re-saves the SAME uploaded image at
    several JPEG quality levels server-side, and reports that the exact
    fingerprint changes completely every time while the perceptual
    fingerprint barely moves. This is the same test proven once via a
    standalone script earlier in the project; this endpoint makes it a
    permanent, repeatable, one-click feature of the app itself.
    """
    if STATE["signer_private_key"] is None:
        raise HTTPException(status_code=503, detail="Signing key unavailable.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    _check_upload_size(file_bytes)

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Couldn't read that image: {exc}") from exc

    original_fingerprint = fingerprint.compute_file_fingerprint(file_bytes)
    original_phash = fingerprint.compute_image_phash(image)

    recompressed_results = []
    for quality in COMPRESSION_TEST_QUALITIES:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        recompressed_bytes = buffer.getvalue()
        recompressed_image = Image.open(io.BytesIO(recompressed_bytes)).convert("RGB")

        recompressed_fingerprint = fingerprint.compute_file_fingerprint(recompressed_bytes)
        recompressed_phash = fingerprint.compute_image_phash(recompressed_image)

        recompressed_results.append({
            "quality": quality,
            "file_size_bytes": len(recompressed_bytes),
            "sha256_merkle_root": recompressed_fingerprint["sha256_merkle_root"],
            "sha256_unchanged": (
                recompressed_fingerprint["sha256_merkle_root"] == original_fingerprint["sha256_merkle_root"]
            ),
            "phash": recompressed_phash,
            "phash_bits_different": fingerprint.hamming_distance(original_phash, recompressed_phash),
        })

    return {
        "original": {
            "file_size_bytes": len(file_bytes),
            "sha256_merkle_root": original_fingerprint["sha256_merkle_root"],
            "phash": original_phash,
        },
        "recompressed": recompressed_results,
        "phash_bits_total": 256,
    }


MAX_RECORD_BYTES = 1 * 1024 * 1024  # 1 MB — a real record is a few hundred bytes to a few KB


@app.post("/api/fingerprint/verify")
async def verify_fingerprint(request: Request):
    """
    Found missing in the round-1 adversarial review: fingerprint.py's
    verify_signature() was written, tested once from a standalone script
    earlier in the project, and then never actually called from the
    running app — every fingerprint the app produced could be SIGNED,
    but the app itself had no way to VERIFY one. Since "anyone can
    independently verify this" is the actual point of Stage 2, that gap
    meant the core claim wasn't demonstrable in the app at all.

    Accepts any record this server produced (from /api/analyze,
    /api/fingerprint/compression-test's "original", or a registry
    record) and checks its signature against this server's OWN public
    key. That's still a meaningful, honest demo — it exercises the exact
    same verify_signature() a real independent verifier would call with
    the public key shown in the footer — it just isn't a substitute for
    an outside party doing the same check with no need to trust this
    server at all, which is what the exposed public key is actually for.

    Takes a raw Request (not a `dict = Body(...)` param) specifically to
    enforce MAX_RECORD_BYTES before parsing — found in round-2 review:
    the first version of this endpoint used Body(...), which has no
    size limit of its own, reopening exactly the unbounded-upload gap
    round 1 had just closed on every file-upload endpoint. A 50 MB JSON
    body was accepted in ~0.65s before this fix. A real record never
    exceeds a few KB, so 1 MB has wide margin without being unbounded.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_RECORD_BYTES:
        raise HTTPException(status_code=413, detail=f"Record too large. Limit is {MAX_RECORD_BYTES} bytes.")

    body = await request.body()
    if len(body) > MAX_RECORD_BYTES:
        raise HTTPException(status_code=413, detail=f"Record too large. Limit is {MAX_RECORD_BYTES} bytes.")

    try:
        record = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc

    if STATE["signer_public_key"] is None:
        raise HTTPException(status_code=503, detail="Signing key unavailable.")

    signature = record.get("signature") if isinstance(record, dict) else None
    if not signature:
        raise HTTPException(status_code=400, detail="Record has no 'signature' field to verify.")

    is_valid = fingerprint.verify_signature(STATE["signer_public_key"], record, signature)
    return {"valid": is_valid}


@app.post("/api/explain/shap-regions")
async def shap_regions(file: UploadFile = File(...)):
    """
    Opt-in region-attribution breakdown (Phase 2, Idea 5a) — deliberately
    a separate endpoint the frontend calls only on request, not part of
    /api/analyze's automatic response. On this model, on CPU, this costs
    several seconds (it re-runs inference ~200 times on masked variants
    of the image); fine for a deliberate "show me why" click, too slow
    to tax on every single upload the way Grad-CAM and the frequency
    panel are fast enough to do.
    """
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail=STATE["load_error"] or "Model is not loaded.")

    extension = Path(file.filename or "").suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Region attribution only supports still images right now, not video.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    _check_upload_size(file_bytes)

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Couldn't read that image: {exc}") from exc

    preprocess = STATE["preprocess"]
    model_view_transform = transforms.Compose(preprocess.transforms[:2])
    model_view_image = model_view_transform(image)

    try:
        segments = shap_explain.segment_superpixels(model_view_image)
        regions = shap_explain.compute_region_shap(score_frames, model_view_image, segments)
    except Exception as exc:  # noqa: BLE001
        logger.exception("SHAP region attribution failed")
        raise HTTPException(status_code=500, detail="Region attribution failed on the server.") from exc

    return {
        "regions": regions,
        "segments_total": int(segments.max()) + 1,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=STATE["load_error"] or "Model is not loaded. Check server logs.",
        )

    extension = Path(file.filename or "").suffix.lower()
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    _check_upload_size(file_bytes)

    started = time.perf_counter()

    # ---------------------------------------------------------------- image
    if extension in IMAGE_EXTENSIONS:
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Couldn't read that image: {exc}") from exc

        try:
            probability_fake = score_frames([image])[0]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed on the server.") from exc

        fingerprint_record = build_fingerprint(
            file_bytes, {"phash": fingerprint.compute_image_phash(image)}
        )
        explainability = build_explainability(image)

        # Only worth asking "which generator" once the image is already
        # believed to be AI-generated — the attribution model was never
        # trained to answer that question about a real photo.
        verdict = "manipulated" if probability_fake >= 0.5 else "authentic"
        generator_attribution = build_generator_attribution(image) if verdict == "manipulated" else None

        return {
            "input_type": "image",
            "filename": file.filename,
            "probability_fake": round(probability_fake, 4),
            "probability_real": round(1.0 - probability_fake, 4),
            "verdict": verdict,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "fingerprint": fingerprint_record,
            "explainability": explainability,
            "generator_attribution": generator_attribution,
        }

    # ---------------------------------------------------------------- video
    if extension in VIDEO_EXTENSIONS:
        try:
            frames, timestamps, meta = sample_frames_from_video(file_bytes, extension)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            probabilities = score_frames(frames)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed on the server.") from exc

        summary = aggregate_video_scores(probabilities)
        headline = summary["probability_fake"]

        # The per-frame timeline is the point of doing this properly:
        # it lets the interface show WHERE in the clip the model saw
        # something, not just a single averaged number.
        timeline = [
            {"t": timestamps[i] if i < len(timestamps) else i, "p": round(p, 4)}
            for i, p in enumerate(probabilities)
        ]

        # Same sampled frames used for AI scoring are reused here for the
        # visual fingerprint — one video-decode pass serves both purposes.
        fingerprint_record = build_fingerprint(
            file_bytes, {"phash_frames": fingerprint.compute_video_phash(frames)}
        )

        # Explainability runs on the single most-suspicious frame only —
        # not all 32 — so cost stays fixed regardless of clip length. It's
        # also the most informative frame to show "why": if anything in
        # the clip triggered the model, this is where.
        peak_index = probabilities.index(max(probabilities))
        explainability = build_explainability(frames[peak_index])
        if explainability is not None:
            explainability["frame_timestamp"] = (
                timestamps[peak_index] if peak_index < len(timestamps) else None
            )

        return {
            "input_type": "video",
            "filename": file.filename,
            "verdict": "manipulated" if headline >= 0.5 else "authentic",
            "probability_real": round(1.0 - headline, 4),
            "timeline": timeline,
            "video": meta,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "fingerprint": fingerprint_record,
            "explainability": explainability,
            **summary,
        }

    supported = sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{extension}'. Supported: {', '.join(supported)}",
    )
