"""
Unit tests for local_features.py -- the distinctive-point evidence behind
the Content Registry's explanations.

The parametrised edit table is the same 18 Mona Lisa edits the README
reports (plus colour edits of the Great Wave, a more colourful painting):
every one must be confirmed by shared points AND described correctly
(area shown, rotation, mirroring, colour). If a future OpenCV, a changed
threshold or a refactor makes any of those facts wrong, this fails.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

import local_features as lf
import registry_explain as rx
from tests.conftest import TEST_IMAGE_FILES, TEST_IMAGES_DIR

ART = Path(__file__).resolve().parents[2] / "demo_artworks"


def _jpeg(im, quality):
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def _centre_crop(im, frac):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    return im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))


def _text(im):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    for i in range(3):
        draw.text((im.width * 0.1, im.height * (0.2 + i * 0.3)), "SAMPLE - NOT FOR SALE", fill=(255, 255, 255))
    return im


def _border(im):
    out = Image.new("RGB", (int(im.width * 1.3), int(im.height * 1.3)), (245, 245, 245))
    out.paste(im, (int(im.width * 0.15), int(im.height * 0.15)))
    return out


def _sepia(im):
    return ImageOps.colorize(ImageOps.grayscale(im), black=(40, 20, 0), white=(255, 240, 200))


# (name, painting, edit, area shown, rotation, mirrored, colour, brightness)
EDITS = [
    ("jpeg q90", "mona_lisa", lambda im: _jpeg(im, 90), 1.0, 0, False, "kept", "same"),
    ("jpeg q10", "mona_lisa", lambda im: _jpeg(im, 10), 1.0, 0, False, "kept", "same"),
    ("shrunk 1/6 then enlarged", "mona_lisa",
     lambda im: im.resize((im.width // 6, im.height // 6)).resize(im.size), 1.0, 0, False, "kept", "same"),
    ("crop centre 80%", "mona_lisa", lambda im: _centre_crop(im, 0.8), 0.64, 0, False, "kept", "same"),
    ("crop centre 60%", "mona_lisa", lambda im: _centre_crop(im, 0.6), 0.36, 0, False, "kept", "same"),
    ("crop centre 40%", "mona_lisa", lambda im: _centre_crop(im, 0.4), 0.16, 0, False, "kept", "same"),
    ("crop centre 25%", "mona_lisa", lambda im: _centre_crop(im, 0.25), 0.0625, 0, False, "kept", "same"),
    ("crop top-left quarter", "mona_lisa",
     lambda im: im.crop((0, 0, im.width // 2, im.height // 2)), 0.25, 0, False, "kept", "same"),
    ("mirror", "mona_lisa", ImageOps.mirror, 1.0, 0, True, "kept", "same"),
    ("rotate 15", "mona_lisa", lambda im: im.rotate(15, expand=True, fillcolor=(255, 255, 255)), 1.0, 15, False, "kept", "same"),
    ("rotate 90", "mona_lisa", lambda im: im.rotate(90, expand=True), 1.0, 90, False, "kept", "same"),
    ("black and white", "mona_lisa", lambda im: ImageOps.grayscale(im).convert("RGB"), 1.0, 0, False, "removed", "same"),
    ("sepia", "mona_lisa", _sepia, 1.0, 0, False, "recoloured", "same"),
    ("brightness +60%", "mona_lisa", lambda im: ImageEnhance.Brightness(im).enhance(1.6), 1.0, 0, False, "kept", "brighter"),
    ("blur radius 4", "mona_lisa", lambda im: im.filter(ImageFilter.GaussianBlur(4)), 1.0, 0, False, "kept", "same"),
    ("text overlay", "mona_lisa", _text, 1.0, 0, False, "kept", "same"),
    ("white border", "mona_lisa", _border, 1.0, 0, False, "kept", "same"),
    ("crop 60% + B&W + jpeg q30", "mona_lisa",
     lambda im: _jpeg(ImageOps.grayscale(_centre_crop(im, 0.6)).convert("RGB"), 30), 0.36, 0, False, "removed", "same"),
    ("great wave sepia", "great_wave", _sepia, 1.0, 0, False, "recoloured", "same"),
    ("great wave black and white", "great_wave", lambda im: ImageOps.grayscale(im).convert("RGB"), 1.0, 0, False, "removed", "same"),
]


def _features(image, max_points):
    frame = lf.examination_frame(image)
    return frame, lf.extract(frame, image.size, max_points)


@pytest.fixture(scope="module")
def references():
    out = {}
    for name in ("mona_lisa", "great_wave"):
        image = Image.open(ART / f"{name}.jpg").convert("RGB")
        out[name] = (image, _features(image, lf.MAX_STORED_POINTS)[1])
    return out


@pytest.mark.parametrize("name,painting,edit,area,rotation,mirrored,colour,brightness", EDITS, ids=[e[0] for e in EDITS])
def test_every_edit_is_confirmed_and_described_correctly(references, name, painting, edit, area, rotation, mirrored,
                                                         colour, brightness):
    original, reference = references[painting]
    frame, query = _features(edit(original), lf.MAX_QUERY_POINTS)
    verification = lf.verify(query, reference, lf.mirror(query, frame, lf.MAX_QUERY_POINTS))

    assert verification.shared_points >= rx.CONFIRMED_POINTS
    geometry = lf.describe_geometry(verification, query, reference)
    assert geometry["original_area_shown"] == pytest.approx(area, abs=0.03)
    assert geometry["rotation_deg"] == pytest.approx(rotation, abs=2)
    assert geometry["mirrored"] is mirrored
    assert geometry["size_ratio"] == pytest.approx(1.0, abs=0.05)
    verdict = lf.compare_colour(frame, verification, reference)
    assert verdict["change"] == colour
    assert verdict["brightness"] == brightness


def test_border_and_rotation_padding_count_as_added_area(references):
    original, reference = references["mona_lisa"]
    for edited, expected in [(_border(original), 1 - 1 / 1.69), (original, 0.0)]:
        frame, query = _features(edited, lf.MAX_QUERY_POINTS)
        verification = lf.verify(query, reference)
        assert lf.describe_geometry(verification, query, reference)["added_area"] == pytest.approx(expected, abs=0.03)


def test_unrelated_images_share_too_few_points_to_confirm(references):
    # The full sweep (594 pairs, max 9 shared points) lives in
    # registry_explanations_test/; this pins the local subset.
    queries = [Image.open(ART / "starry_night.jpg").convert("RGB")]
    queries += [Image.open(TEST_IMAGES_DIR / f).convert("RGB") for f in TEST_IMAGE_FILES]
    for _, reference in references.values():
        for image in queries:
            frame, query = _features(image, lf.MAX_QUERY_POINTS)
            shared = lf.verify(query, reference, lf.mirror(query, frame, lf.MAX_QUERY_POINTS)).shared_points
            assert shared < rx.CONFIRMED_POINTS


def test_flat_image_has_no_points_and_does_not_crash(references):
    flat = Image.new("RGB", (400, 300), (128, 128, 128))
    frame, query = _features(flat, lf.MAX_QUERY_POINTS)
    assert query.count == 0
    verification = lf.verify(query, references["mona_lisa"][1], lf.mirror(query, frame, lf.MAX_QUERY_POINTS))
    assert verification.shared_points == 0
    assert verification.transform is None


def test_large_upload_is_examined_at_bounded_size():
    big = Image.new("RGB", (4000, 3000), (10, 20, 30))
    assert max(lf.examination_frame(big).shape[:2]) == lf.FRAME_LONG_SIDE
    small = Image.new("RGB", (300, 200))
    assert lf.examination_frame(small).shape[:2] == (200, 300)  # never enlarged


def test_features_survive_a_storage_round_trip(references):
    stored = references["mona_lisa"][1]
    restored = lf.LocalFeatures.from_arrays(stored.to_arrays())
    assert restored.frame_size == stored.frame_size
    assert restored.source_size == stored.source_size
    assert np.array_equal(restored.descriptors, stored.descriptors)
    assert np.allclose(restored.points, stored.points)


def test_palette_is_deterministic_and_shares_sum_to_one(references):
    frame = lf.examination_frame(references["great_wave"][0])
    first, second = lf.palette(frame), lf.palette(frame)
    assert first == second
    assert sum(c["share"] for c in first) == pytest.approx(1.0, abs=0.01)
    assert all(c["hex"].startswith("#") and len(c["hex"]) == 7 for c in first)


def test_area_index_covers_the_grid_and_rejects_outside_points():
    assert lf.area_index(0.0, 0.0) == 0
    assert lf.area_index(0.5, 0.5) == 4
    assert lf.area_index(0.99, 0.99) == 8
    assert lf.area_index(1.0, 0.5) is None
    assert lf.area_index(-0.1, 0.5) is None
