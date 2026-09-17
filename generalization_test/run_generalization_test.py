"""
Deep-Guard -- Generator Attribution: cross-generator generalization test.

Standalone script, run by hand, not part of the live app -- exact same
pattern as bias_map/build_bias_map.py. Answers the question the Phase 2
plan called "the generalization test that matters most" and flagged as
never having been done: what does the attribution model actually say
when shown a real image from a generator it never trained on?

held_out_images/ contains real images from two generators genuinely
outside GENERATOR_CLASSES -- midjourney (current showcase, gathered
from midjourney.com/showcase) and dalle3 (OpenAI's own official DALL-E
3 examples page) -- both first-party sources, not scraped from a
third party claiming attribution. See labeled_test_set.csv for the
exact file -> generator mapping.

The attribution model CANNOT say "I don't know" -- it is a softmax
over exactly 8 fixed classes. Every image below gets a confident,
wrong answer by construction. What matters is HOW it's wrong:
  - Is confidence appropriately lower than on a real training-
    distribution image, or just as falsely confident?
  - Is the wrong guess at least "family-coherent" (both midjourney and
    dalle3 are diffusion-family tools -- does the model at least land
    on another diffusion class, suggesting it learned something
    transferable) or does it scatter onto GAN classes too (suggesting
    no real transferable signal)?

Also scores every image through the Stage 1 base detector -- in the
real app, attribution only ever runs downstream of Stage 1 flagging
something as fake, so whether Stage 1 even catches these unseen-
generator images in the first place is a directly relevant, separate
finding.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

APP_DIR = Path(__file__).resolve().parents[1] / "deepguard-bouncer" / "app"
sys.path.insert(0, str(APP_DIR))

import attribution  # noqa: E402
from model import GENERATOR_CLASSES, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "held_out_images"
MANIFEST_PATH = ROOT / "labeled_test_set.csv"
RESULTS_PATH = ROOT / "results.json"

DETECTOR_CHECKPOINT_PATH = APP_DIR.parent / "models" / "deepguard_bouncer.pth"
ATTRIBUTION_CHECKPOINT_PATH = APP_DIR.parent / "models" / "deepguard_attribution.pth"

DEVICE = torch.device("cpu")

# Both held-out tools are diffusion-based -- Midjourney's own published
# architecture is a latent diffusion model, and DALL-E 3 is OpenAI's
# diffusion-based successor to DALL-E 2. Used only to check whether a
# wrong guess at least lands in the right family.
GENERATOR_FAMILY = {
    "stylegan2": "gan", "pro_gan": "gan", "big_gan": "gan", "cycle_gan": "gan",
    "ddpm": "diffusion", "latent_diffusion": "diffusion",
    "stable_diffusion": "diffusion", "glide": "diffusion",
}
TRUE_FAMILY = {"midjourney": "diffusion", "dalle3": "diffusion"}


def load_stage1_detector():
    checkpoint = torch.load(DETECTOR_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"model_state_dict": checkpoint}

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])
    input_size = checkpoint.get("input_size", 224)
    resize_to = int(input_size * 256 / 224)
    preprocess = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, preprocess


def score_stage1(model, preprocess, image: Image.Image) -> float:
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor)
        return float(torch.sigmoid(logit).item())


def main():
    print("Loading Stage 1 base detector...")
    detector_model, detector_preprocess = load_stage1_detector()

    print("Loading attribution model...")
    attribution_state = attribution.load_attribution_model(ATTRIBUTION_CHECKPOINT_PATH, DEVICE)
    if attribution_state is None:
        print(f"FATAL: no attribution checkpoint at {ATTRIBUTION_CHECKPOINT_PATH}")
        sys.exit(1)
    assert attribution_state["class_order"] == GENERATOR_CLASSES, "class_order mismatch -- checkpoint/model.py drifted apart"

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded manifest: {len(rows)} images")

    per_image = []
    per_generator = defaultdict(list)

    for row in rows:
        true_generator = row["true_generator"]
        image_path = IMAGES_DIR / true_generator / row["filename"]
        image = Image.open(image_path).convert("RGB")

        stage1_prob_fake = score_stage1(detector_model, detector_preprocess, image)
        attr_result = attribution.predict_generator(attribution_state, image, DEVICE)

        predicted = attr_result["predicted_generator"]
        confidence = attr_result["probabilities"][predicted]
        family_match = GENERATOR_FAMILY[predicted] == TRUE_FAMILY[true_generator]

        record = {
            "filename": row["filename"],
            "true_generator": true_generator,
            "stage1_probability_fake": round(stage1_prob_fake, 4),
            "stage1_would_flag_manipulated_50pct": stage1_prob_fake >= 0.5,
            "stage1_would_show_manipulated_headline_80pct": stage1_prob_fake >= 0.8,
            "attribution_predicted_generator": predicted,
            "attribution_confidence": confidence,
            "attribution_family_coherent": family_match,
        }
        per_image.append(record)
        per_generator[true_generator].append(record)
        print(f"  {row['filename']:30s} stage1={stage1_prob_fake:.3f}  attribution={predicted} ({confidence:.3f})  family_ok={family_match}")

    summary_by_generator = {}
    for gen, records in per_generator.items():
        n = len(records)
        summary_by_generator[gen] = {
            "n": n,
            "stage1_catch_rate_50pct": round(sum(r["stage1_would_flag_manipulated_50pct"] for r in records) / n, 3),
            "stage1_catch_rate_80pct_headline": round(sum(r["stage1_would_show_manipulated_headline_80pct"] for r in records) / n, 3),
            "attribution_mean_confidence": round(sum(r["attribution_confidence"] for r in records) / n, 3),
            "attribution_family_coherence_rate": round(sum(r["attribution_family_coherent"] for r in records) / n, 3),
            "attribution_predicted_label_distribution": dict(Counter(r["attribution_predicted_generator"] for r in records)),
        }

    overall_confidence = sum(r["attribution_confidence"] for r in per_image) / len(per_image)
    overall_family_coherence = sum(r["attribution_family_coherent"] for r in per_image) / len(per_image)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": len(per_image),
        "generators_tested": sorted(per_generator.keys()),
        "note": (
            "Every image here is from a generator OUTSIDE GENERATOR_CLASSES "
            "(the 8 classes the attribution model was trained on). The model "
            "cannot say 'unknown' -- it is a fixed 8-class softmax -- so every "
            "prediction below is necessarily wrong by construction. What's "
            "being measured is HOW it's wrong: whether confidence drops "
            "appropriately on unfamiliar input, and whether the wrong guess "
            "at least lands in the right generator family (GAN vs diffusion)."
        ),
        "overall_attribution_mean_confidence": round(overall_confidence, 3),
        "overall_attribution_family_coherence_rate": round(overall_family_coherence, 3),
        "by_generator": summary_by_generator,
        "per_image": per_image,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print(f"\nOverall attribution mean confidence on unseen generators: {overall_confidence:.3f}")
    print(f"Overall family-coherence rate: {overall_family_coherence:.3f}")
    for gen, s in summary_by_generator.items():
        print(f"\n{gen} (n={s['n']}):")
        print(f"  Stage 1 catch rate (>=50%): {s['stage1_catch_rate_50pct']}")
        print(f"  Stage 1 catch rate (>=80%, real headline threshold): {s['stage1_catch_rate_80pct_headline']}")
        print(f"  Attribution mean confidence: {s['attribution_mean_confidence']}")
        print(f"  Attribution family-coherence rate: {s['attribution_family_coherence_rate']}")
        print(f"  Predicted label distribution: {s['attribution_predicted_label_distribution']}")


if __name__ == "__main__":
    main()
