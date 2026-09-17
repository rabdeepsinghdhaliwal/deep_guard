"""
Deep-Guard -- Bias Map builder (Phase 2, Idea 1).

Standalone script, NOT part of the running app -- run this by hand
whenever labeled_test_set.csv or the model changes:

    cd deepguard-bouncer/bias_map
    ../venv/Scripts/python build_bias_map.py

It loads the same checkpoint main.py loads, scores every image in
labeled_test_set.csv, groups the results by generator / subject / style
(individually and in combination), and writes results.json -- which
main.py's GET /api/bias-map endpoint just reads back, unchanged. Nothing
here runs on a live request, so it doesn't need to be fast.

HONEST SCOPE NOTE, read before trusting any number this produces: the
shipped labeled_test_set.csv has only the project's original 8 test
images. That's enough to prove this machinery works end-to-end, and
it's exactly reused rather than re-invented -- but 8 images is not a
statistically meaningful bias map, and none of them have a KNOWN
generator (StyleGAN vs. Stable Diffusion vs. ...) attached, because
nobody told this project which tool made them -- they're all recorded
as "unknown" rather than guessed. Building the real version -- 300-500
images with genuine generator labels -- needs either a properly
generator-labeled dataset (ArtiFact, per the Phase 2 doc) or manually
curated/labeled images, which is real, un-automatable human work. This
script's output should be read as "the dashboard works," not as "here
is Deep-Guard's bias profile."
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from model import build_model  # noqa: E402 -- must follow the sys.path insert above

BIAS_MAP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BIAS_MAP_DIR.parent.parent
MODEL_PATH = BIAS_MAP_DIR.parent / "models" / "deepguard_bouncer.pth"
CSV_PATH = BIAS_MAP_DIR / "labeled_test_set.csv"
IMAGE_DIR = PROJECT_ROOT / "test_images"
RESULTS_PATH = BIAS_MAP_DIR / "results.json"

# Groups this small are noise, not signal -- flagged in the output
# rather than silently presented with the same confidence as a
# well-sampled group.
MIN_TRUSTWORTHY_GROUP_SIZE = 20


def load_model_for_scoring():
    """Deliberately duplicated from main.py's load_model() rather than
    imported -- this script has no FastAPI app or STATE dict to hang a
    shared function off of, and the duplication is small and obvious
    enough to keep in sync by inspection (unlike model.py's actual
    architecture, which is the thing that would silently corrupt
    results if it drifted, and stays imported, not copied)."""
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"model_state_dict": checkpoint}

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_size = checkpoint.get("input_size", 224)
    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])
    resize_to = int(input_size * 256 / 224)
    preprocess = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, preprocess, device


def score_image(model, preprocess, device, image_path: Path) -> float:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        return torch.sigmoid(logit).item()


def group_accuracy(rows: list[dict], key_fn) -> dict:
    """Groups `rows` by key_fn(row) and returns {group_key: {accuracy,
    sample_count, trustworthy}} for each group."""
    groups: dict = {}
    for row in rows:
        key = key_fn(row)
        groups.setdefault(key, []).append(row)

    result = {}
    for key, group_rows in groups.items():
        correct = sum(1 for r in group_rows if r["correct"])
        n = len(group_rows)
        result[key] = {
            "accuracy": round(correct / n, 4),
            "sample_count": n,
            "trustworthy": n >= MIN_TRUSTWORTHY_GROUP_SIZE,
        }
    return result


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"No model found at {MODEL_PATH} -- nothing to score against.")
    if not CSV_PATH.exists():
        raise SystemExit(f"No labeled test set found at {CSV_PATH}.")

    model, preprocess, device = load_model_for_scoring()

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            image_path = IMAGE_DIR / record["filename"]
            if not image_path.exists():
                print(f"  [skip] {record['filename']} not found at {image_path}")
                continue

            probability_fake = score_image(model, preprocess, device, image_path)
            predicted_label = "fake" if probability_fake >= 0.5 else "real"
            correct = predicted_label == record["true_label"]

            rows.append({
                **record,
                "probability_fake": round(probability_fake, 4),
                "predicted_label": predicted_label,
                "correct": correct,
            })
            print(f"  {record['filename']:<14} true={record['true_label']:<5} "
                  f"predicted={predicted_label:<5} p_fake={probability_fake:.4f} "
                  f"{'OK' if correct else 'WRONG'}")

    overall_correct = sum(1 for r in rows if r["correct"])
    overall_accuracy = round(overall_correct / len(rows), 4) if rows else 0.0

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": len(rows),
        "overall_accuracy": overall_accuracy,
        "min_trustworthy_group_size": MIN_TRUSTWORTHY_GROUP_SIZE,
        "scope_warning": (
            f"The numbers below are not yet meaningful -- please don't read anything into "
            f"them. This run used only {len(rows)} images (the project's original test set), "
            f"far below the {MIN_TRUSTWORTHY_GROUP_SIZE}+ needed per group before an accuracy "
            f"number means anything, and generator is 'unknown' for every one of them because "
            f"nobody recorded which AI tool made them. What this run DOES prove: the pipeline "
            f"-- score every image, slice by category, render as a dashboard -- works "
            f"correctly end to end. Turning it into a real bias map needs a properly sized, "
            f"generator-labeled dataset (300-500 images; see the Phase 2 build plan) in place "
            f"of this placeholder one."
        ),
        "by_generator": group_accuracy(rows, lambda r: r["generator"]),
        "by_subject": group_accuracy(rows, lambda r: r["subject"]),
        "by_style": group_accuracy(rows, lambda r: r["style"]),
        "by_combination": group_accuracy(rows, lambda r: f"{r['generator']} / {r['subject']} / {r['style']}"),
        "rows": rows,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nOverall accuracy: {overall_accuracy:.1%} ({overall_correct}/{len(rows)})")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
