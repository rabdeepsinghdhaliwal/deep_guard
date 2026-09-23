"""
Unit tests for registry_explain.py and the explanation parts of
content_registry.py, run against the real SSCD model.

The central promise these pin: the per-thing "share of the similarity"
is an EXACT split of the registry's own score -- the full set of cells
reproduces the reported similarity, and the shares add back up to it --
so "what was responsible for the detection" is answered by the network
that made the detection.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

import content_registry
import local_features as lf
import main
import registry_explain as rx

ART = Path(__file__).resolve().parents[2] / "demo_artworks"


def _jpeg(im, quality):
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def _centre_crop(im, frac):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    return im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))


@pytest.fixture(scope="module")
def sscd():
    return content_registry.load_sscd_model(main.SSCD_MODEL_PATH)


@pytest.fixture(scope="module")
def mona_lisa():
    return Image.open(ART / "mona_lisa.jpg").convert("RGB")


@pytest.fixture(scope="module")
def registered(sscd, mona_lisa):
    """The Mona Lisa, described and stored the way a registration stores it
    (no namer, so these tests don't depend on the optional CLIP files)."""
    upload = rx.describe(mona_lisa, None, lf.MAX_STORED_POINTS)
    return {"stored": rx.storable(upload, None), "vector": content_registry.compute_content_embedding(sscd, mona_lisa)}


def _explain(sscd, registered, query_image, labeler=None, stored="default"):
    upload = rx.describe(query_image, labeler, lf.MAX_QUERY_POINTS)
    vector, feature_map = content_registry.compute_content_features(sscd, query_image)
    similarity = float(vector @ registered["vector"])
    stored_arrays = registered["stored"] if stored == "default" else stored
    return similarity, rx.explain_match(upload, similarity, stored_arrays, sscd, registered["vector"], feature_map)


def test_features_path_reproduces_the_plain_embedding(sscd, mona_lisa):
    vector, feature_map = content_registry.compute_content_features(sscd, mona_lisa)
    assert np.allclose(vector, content_registry.compute_content_embedding(sscd, mona_lisa), atol=1e-6)
    assert feature_map.shape[1] == 2048


def test_similarity_shares_are_an_exact_split_of_the_score(sscd, registered, mona_lisa):
    crop = _centre_crop(mona_lisa, 0.6)
    vector, feature_map = content_registry.compute_content_features(sscd, crop)
    cells = feature_map.shape[2] * feature_map.shape[3]
    owners = [i % 4 for i in range(cells)]  # any grouping must split exactly
    shares, total = content_registry.similarity_shares(sscd, feature_map, registered["vector"], owners)
    assert total == pytest.approx(float(vector @ registered["vector"]), abs=1e-4)
    assert sum(shares.values()) == pytest.approx(total, abs=1e-4)


def test_combo_edit_is_explained_with_facts_and_exact_shares(sscd, registered, mona_lisa):
    combo = _jpeg(ImageOps.grayscale(_centre_crop(mona_lisa, 0.6)).convert("RGB"), 30)
    similarity, explanation = _explain(sscd, registered, combo)
    assert explanation["available"] and explanation["confirmed"]
    assert explanation["strength"] == "strong"
    assert explanation["geometry"]["original_area_shown"] == pytest.approx(0.36, abs=0.03)
    assert explanation["geometry"]["shown_position"] == "centre"
    assert explanation["colour"]["change"] == "removed"
    total = sum(t["share"] for t in explanation["common"]) + explanation["not_from_original_share"]
    assert total == pytest.approx(similarity, abs=1e-3)
    assert explanation["figure"]["lines"] and len(explanation["figure"]["footprint"]) == 4


def test_top_left_crop_lists_what_was_cropped_away(sscd, registered, mona_lisa):
    corner = mona_lisa.crop((0, 0, mona_lisa.width // 2, mona_lisa.height // 2))
    _, explanation = _explain(sscd, registered, corner)
    assert explanation["confirmed"]
    missing = {p for t in explanation["only_in_original"] for p in t["positions"]}
    # Without a namer each area is its own thing: the bottom row can't be in a top-left quarter.
    assert {"bottom left", "bottom centre", "bottom right"} <= missing


def test_added_border_gets_a_negative_share(sscd, registered, mona_lisa):
    bordered = Image.new("RGB", (int(mona_lisa.width * 1.3), int(mona_lisa.height * 1.3)), (245, 245, 245))
    bordered.paste(mona_lisa, (int(mona_lisa.width * 0.15), int(mona_lisa.height * 0.15)))
    _, explanation = _explain(sscd, registered, bordered)
    assert explanation["geometry"]["added_area"] == pytest.approx(0.41, abs=0.03)
    assert explanation["not_from_original_share"] < 0


def test_unconfirmed_match_splits_the_score_over_the_uploads_own_areas(sscd, registered):
    unrelated = Image.open(ART / "starry_night.jpg").convert("RGB")
    similarity, explanation = _explain(sscd, registered, unrelated)
    assert explanation["available"] and not explanation["confirmed"]
    assert explanation["strength"] in {"none", "weak"}
    assert sum(p["share"] for p in explanation["upload_parts"]) == pytest.approx(similarity, abs=1e-3)


def test_entry_without_stored_features_is_unavailable_not_an_error(sscd, registered, mona_lisa):
    _, explanation = _explain(sscd, registered, mona_lisa, stored=None)
    assert explanation["available"] is False
    assert "registered before" in explanation["reason"]


def test_without_a_namer_areas_are_named_by_position(mona_lisa):
    upload = rx.describe(mona_lisa, None, lf.MAX_STORED_POINTS)
    names = [t["name"] for t in upload.things]
    assert len(names) == 9 and "top left area" in names
    assert upload.summary["medium"] is None and upload.summary["namer"] is None


def test_with_the_namer_the_mona_lisa_gets_content_names(mona_lisa):
    if main.STATE["concept_labeler"] is None:
        labeler = main.concept_labels.load_concept_labeler(main.CONCEPT_LABELER_ENCODER_PATH, main.CONCEPT_LABELER_VOCAB_PATH)
    else:
        labeler = main.STATE["concept_labeler"]
    if labeler is None:
        pytest.skip("namer not built on this machine (tools/build_concept_labeler.py)")
    upload = rx.describe(mona_lisa, labeler, lf.MAX_STORED_POINTS)
    names = {t["name"] for t in upload.things}
    assert "face" in names and "landscape" in names
    assert upload.summary["medium"]["name"] == "oil painting"


def test_stale_features_file_is_never_attached_to_a_new_entry(tmp_path, registered):
    registry = content_registry.ContentRegistry(tmp_path / "r.index", tmp_path / "r.json", features_dir=tmp_path / "f")
    first = {"title": "A", "sha256_merkle_root": "aaa", "timestamp_utc": "t1"}
    entry = registry.register(registered["vector"], first, registered["stored"])
    assert registry.load_features(entry) is not None
    # Simulate "delete the registry files, keep the features folder, register something else"
    (tmp_path / "r.index").unlink()
    (tmp_path / "r.json").unlink()
    fresh = content_registry.ContentRegistry(tmp_path / "r.index", tmp_path / "r.json", features_dir=tmp_path / "f")
    other = fresh.register(registered["vector"], {"title": "B", "sha256_merkle_root": "bbb", "timestamp_utc": "t2"})
    assert other == entry  # same slot number...
    assert fresh.load_features(other) is None  # ...but the old file is not treated as B's
