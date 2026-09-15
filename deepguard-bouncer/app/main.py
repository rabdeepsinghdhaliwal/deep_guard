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
"""

import io
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

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
            f"No model file found at {MODEL_PATH}. Train the model in "
            f"Colab, download deepguard_bouncer.pth, and place it at "
            f"exactly that path. See README.md, Part 4."
        )
        logger.error(STATE["load_error"])
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
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
    }


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
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) / 1e6:.1f} MB). Limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )

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

        return {
            "input_type": "image",
            "filename": file.filename,
            "probability_fake": round(probability_fake, 4),
            "probability_real": round(1.0 - probability_fake, 4),
            "verdict": "manipulated" if probability_fake >= 0.5 else "authentic",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
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

        return {
            "input_type": "video",
            "filename": file.filename,
            "verdict": "manipulated" if headline >= 0.5 else "authentic",
            "probability_real": round(1.0 - headline, 4),
            "timeline": timeline,
            "video": meta,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            **summary,
        }

    supported = sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{extension}'. Supported: {', '.join(supported)}",
    )
