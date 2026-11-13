"""
How well the hidden ID survives, measured on every image in the repository.

For each image (the 8 labelled test photos, the 44 held-out DALL-E 3 and
Midjourney images, the 3 demo paintings) this makes the credentialed copy
exactly as registration does (EXIF rotation applied, JPEG quality 95 or PNG),
then puts copies of it through everyday damage and reads the ID back:

    untouched        the credentialed copy as downloaded
    whatsapp         long side at most 1600 px, JPEG quality 75
    half_q70         half the size, JPEG quality 70
    crop80_whatsapp  centre crop keeping 80% of the area, then whatsapp
    screenshot       75% size, saved as PNG
    greyscale        colour removed, JPEG quality 85

It also reads every UNMARKED original, to count IDs "found" by chance, and
measures how visible the mark is (PSNR, higher is less visible; 40 dB and up
is generally invisible). Each image gets the same ID on every run (derived
from its file name), so the numbers are reproducible.

    python tools/measure_watermark.py            (writes measurements/watermark_robustness.json)
"""

import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parent
REPO = LEDGER.parent
sys.path.insert(0, str(LEDGER))
import watermark  # noqa: E402

OUT = LEDGER / "measurements" / "watermark_robustness.json"


def images() -> list[Path]:
    found = sorted((REPO / "test_images").glob("ai[0-9].png")) + sorted((REPO / "test_images").glob("real[0-9].*"))
    found += sorted((REPO / "generalization_test" / "held_out_images").rglob("*.webp"))
    found += sorted((REPO / "deepguard-bouncer" / "demo_artworks").glob("*.jpg"))
    return found


def fixed_id(path: Path) -> str:
    n = int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "big") >> 3  # 61 bits
    return format(n, f"0{watermark.ID_BITS}b")


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def png(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def whatsapp(image: Image.Image) -> Image.Image:
    scale = min(1.0, 1600 / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return jpeg(image, 75)


def crop80(image: Image.Image) -> Image.Image:
    side = 0.8 ** 0.5
    w, h = image.size
    cw, ch = round(w * side), round(h * side)
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


CONDITIONS = {
    "untouched": lambda im: im,
    "whatsapp": whatsapp,
    "half_q70": lambda im: jpeg(im.resize((im.width // 2, im.height // 2), Image.LANCZOS), 70),
    "crop80_whatsapp": lambda im: whatsapp(crop80(im)),
    "screenshot": lambda im: png(im.resize((round(im.width * .75), round(im.height * .75)), Image.LANCZOS)),
    "greyscale": lambda im: jpeg(ImageOps.grayscale(im).convert("RGB"), 85),
}


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main() -> int:
    marker = watermark.Watermarker()
    marker.warm_up()
    if not marker.available:
        print(f"TrustMark unavailable: {marker.load_error}")
        return 2
    paths = images()
    rows, started = [], time.time()
    for path in paths:
        original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        bits = fixed_id(path)
        marked = marker.embed(original, bits)
        copy = png(marked) if path.suffix.lower() == ".png" else jpeg(marked, 95)
        row = {"image": str(path.relative_to(REPO)).replace("\\", "/"), "size": list(original.size),
               "psnr_db": round(psnr(original, copy), 1),
               "chance_id_in_unmarked_original": marker.read(original) is not None}
        for name, damage in CONDITIONS.items():
            row[name] = marker.read(damage(copy)) == bits
        rows.append(row)
        print(f"  {row['image']:<62} {row['psnr_db']:>5} dB  " + " ".join(
            ("ok" if row[name] else "--") for name in CONDITIONS))
    n = len(rows)
    summary = {name: f"{sum(r[name] for r in rows)}/{n}" for name in CONDITIONS}
    large = [r for r in rows if min(r["size"]) >= 512]
    summary["half_q70_short_side_at_least_512px"] = f"{sum(r['half_q70'] for r in large)}/{len(large)}"
    summary["chance_ids_in_unmarked_originals"] = f"{sum(r['chance_id_in_unmarked_original'] for r in rows)}/{n}"
    summary["psnr_db_median"] = round(float(np.median([r["psnr_db"] for r in rows])), 1)
    summary["psnr_db_lowest"] = min(r["psnr_db"] for r in rows)
    print("\nID read back correctly:")
    for key, value in summary.items():
        print(f"  {key:<38} {value}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "watermark": {"alg": watermark.ALG, "schema": watermark.SCHEMA, "id_bits": watermark.ID_BITS},
        "conditions": {name: CONDITIONS[name].__doc__ or name for name in CONDITIONS},
        "seconds": round(time.time() - started, 1), "summary": summary, "per_image": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
