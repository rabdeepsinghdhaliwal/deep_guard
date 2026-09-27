"""
Tests for the evaluation tools themselves -- the scorer is only worth
trusting if its own arithmetic is checked. Pure logic, no downloads, no
model weights; runs in about a second:

    cd deepguard-bouncer
    venv/Scripts/python -m pytest evaluation/test_evaluation_tools.py -q
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_exam_set as build  # noqa: E402
import run_exam  # noqa: E402


# ------------------------------------------------------------ build_exam_set --

def test_pick_row_groups_fetches_the_rare_generator_first():
    index = ([{"shard": "s", "row_group": 0, "row": i, "model": "common"} for i in range(50)]
             + [{"shard": "s", "row_group": 1, "row": i, "model": "common"} for i in range(49)]
             + [{"shard": "s", "row_group": 1, "row": 49, "model": "rare"}])
    chosen = build.pick_row_groups(index, lambda r: r["model"], {"rare": 1, "common": 5}, max_groups=1)
    assert chosen == [("s", 1)]  # the only group holding the rare one, and it also covers "common"


def test_pick_row_groups_stops_when_quotas_are_met():
    index = [{"shard": "s", "row_group": g, "row": 0, "model": "m"} for g in range(10)]
    assert len(build.pick_row_groups(index, lambda r: r["model"], {"m": 3}, max_groups=10)) == 3


def _png(color, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_save_keeps_original_bytes_and_names_by_true_format(tmp_path):
    data = _png("red")
    rel, facts = build.save(tmp_path, "sub", "odd name:with/colon", data)
    assert rel.endswith(".png") and ":" not in rel and " " not in rel
    assert (tmp_path / rel).read_bytes() == data
    assert facts["sha256"] == build.sha256(data) and (facts["width"], facts["height"]) == (64, 48)
    assert build.save(tmp_path, "sub", "odd name:with/colon", data)[0] == rel  # idempotent


def test_save_refuses_to_reuse_a_name_for_a_different_image(tmp_path):
    build.save(tmp_path, "sub", "same", _png("red"))
    with pytest.raises(SystemExit, match="collision"):
        build.save(tmp_path, "sub", "same", _png("blue"))


def test_social_copy_is_a_smaller_jpeg():
    big = _png("green", size=(3000, 2000))
    out = Image.open(io.BytesIO(build.social_copy(big)))
    assert out.format == "JPEG" and max(out.size) == 1600


def test_generator_groups():
    assert build.generator_group("openfake-core-test", "", "real") == "real photo"
    assert build.generator_group("openfake-core-test", "gpt-image-2", "ai") == "closed commercial"
    assert build.generator_group("openfake-core-test", "veo-3", "ai") == "video-model frame"
    assert build.generator_group("cnndetection", "stylegan2", "ai") == "older research model"
    assert build.generator_group("genimage", "midjourney_imagenet", "ai") == "closed commercial"
    assert build.generator_group("wikimedia-commons", "stable-diffusion (version not stated)", "ai") == "open weights"


# ------------------------------------------------------------------ run_exam --

def test_page_verdict_matches_the_analysis_page_thresholds():
    assert run_exam.page_verdict(0.80) == "ai"
    assert run_exam.page_verdict(0.7999) == "inconclusive"
    assert run_exam.page_verdict(0.20) == "real"
    assert run_exam.page_verdict(0.2001) == "inconclusive"


def test_wilson_interval_contains_the_rate_and_stays_in_range():
    lo, hi = run_exam.wilson(3, 10)
    assert 0 <= lo < 0.3 < hi <= 1
    assert run_exam.wilson(0, 0) == (None, None)
    assert run_exam.wilson(0, 20)[0] == 0.0


def _rec(label, p, **kw):
    return {"label": label, "p_ai": p, "logit": run_exam.math.log(max(p, 1e-9) / max(1 - p, 1e-9)),
            "verdict": run_exam.page_verdict(p), **kw}


def test_outcome_counts_add_up():
    rows = [_rec("ai", 0.95), _rec("ai", 0.5), _rec("ai", 0.05), _rec("real", 0.1), _rec("real", 0.9)]
    c = run_exam.outcome_counts(rows)
    assert (c["ai_called_ai"], c["ai_inconclusive"], c["ai_called_real"]) == (1, 1, 1)
    assert (c["real_called_real"], c["real_called_ai"]) == (1, 1)
    assert c["correct"] + c["wrong"] + c["inconclusive"] == pytest.approx(1.0)
    assert c["trustworthy"] is False  # 5 images < MIN_GROUP


def test_unknown_tool_test_leaves_out_images_whose_tool_was_never_recorded():
    base = dict(set="calibration", attr_top="glide", attr_energy=1.0)
    rows = [
        {**base, "label": "ai", "attribution_class": "glide", "generator": "glide_imagenet", "attr_top_p": 0.9},
        {**base, "label": "ai", "attribution_class": "", "generator": "gpt-image-2", "attr_top_p": 0.3},
        {**base, "label": "ai", "attribution_class": "", "generator": "unknown", "attr_top_p": 0.99},
        {**base, "label": "ai", "attribution_class": "", "generator": "stylegan (version not stated)", "attr_top_p": 0.99},
    ]
    res = run_exam.attribution_section(rows, rows)
    assert res["calibration"] == {"known": 1, "unknown": 1}
    # Separable here, so the rule picks the lowest bar that names no unknown tool.
    assert res["unknown_threshold"] == 0.3


def test_incoming_folders_turn_your_own_pictures_into_labelled_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "EXAM_DIR", tmp_path)
    build.incoming_part(set())                      # first run: creates the folders and their notes
    assert (tmp_path / "incoming" / "phone_real" / "WHAT_GOES_HERE.txt").exists()
    (tmp_path / "incoming" / "phone_real" / "IMG_0001.png").write_bytes(_png("gray"))
    (tmp_path / "incoming" / "whatsapp_ai" / "IMG-WA0002.png").write_bytes(_png("purple"))
    rows = {r["source"]: r for r in build.incoming_part(set())}
    assert rows["user-phone_real"]["label"] == "real" and rows["user-phone_real"]["purpose"] == "detector"
    assert rows["user-whatsapp_ai"]["label"] == "ai"
    assert rows["user-whatsapp_ai"]["transform"] == "whatsapp round trip"
    assert rows["user-phone_real"]["sha256"] == build.sha256(_png("gray"))


def _att(p, known):
    return {"attr_top_p": p, "attribution_class": "glide" if known else ""}


def test_naming_is_switched_off_when_confidence_does_not_separate_known_from_unknown():
    # The real situation: unknown tools get HIGHER top probabilities (AUROC < 0.5).
    known = [_att(p, True) for p in (0.40, 0.55, 0.60, 0.70)]
    unknown = [_att(p, False) for p in (0.80, 0.90, 0.95, 0.99, 0.65)]
    threshold, why = run_exam.choose_unknown_threshold(known, unknown)
    assert threshold == 1.0 and "switched off" in why


def test_when_it_does_separate_the_rule_caps_unknown_naming_at_five_percent():
    known = [_att(0.9 + i / 1000, True) for i in range(20)]
    unknown = [_att(i / 100, False) for i in range(40)]          # 0.00 .. 0.39
    threshold, _ = run_exam.choose_unknown_threshold(known, unknown)
    named = sum(r["attr_top_p"] > threshold for r in unknown) / len(unknown)
    assert named <= run_exam.MAX_UNKNOWN_NAMED
    assert all(r["attr_top_p"] > threshold for r in known)       # and it still names every known tool


def test_whatsapp_copies_are_paired_with_their_originals_by_file_name(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "EXAM_DIR", tmp_path)
    build.incoming_part(set())
    (tmp_path / "incoming" / "phone_real" / "IMG_7.png").write_bytes(_png("gray"))
    (tmp_path / "incoming" / "whatsapp_real" / "IMG_7.png").write_bytes(_png("silver"))   # same name
    (tmp_path / "incoming" / "whatsapp_real" / "IMG_8.png").write_bytes(_png("white"))    # no original
    rows = {r["id"]: r for r in build.incoming_part(set())}
    copy = rows["user-whatsapp_real-IMG_7"]
    assert copy["purpose"] == "robustness" and copy["variant_of"] == "user-phone_real-IMG_7"
    lone = rows["user-whatsapp_real-IMG_8"]
    assert lone["purpose"] == "detector" and lone["variant_of"] == "" and "no original" in lone["notes"]


def test_percentages_round_half_up_like_the_bias_map_page():
    assert run_exam.pct(0.125) == "13%" and run_exam.pct(0.135) == "14%" and run_exam.pct(None) == "—"


# ------------------------------------------------------------------ snapshot --

import snapshot  # noqa: E402


def test_snapshot_compare_catches_changed_values_not_just_changed_fields():
    a = {"x": {"heatmap_png_base64": {"sha256": "aaa"}, "probability_fake": 0.5,
               "explainability": {"model_consistency": {"std_probability": 0.01}}, "timestamp_utc": "t1"}}
    b = {"x": {"heatmap_png_base64": {"sha256": "bbb"}, "probability_fake": 0.5001,
               "explainability": {"model_consistency": {"std_probability": 0.04}}, "timestamp_utc": "t2"}}
    diffs = {path: verdict for path, _, _, verdict in snapshot.value_diffs(a, b)}
    assert diffs[".x.heatmap_png_base64.sha256"] == "CHANGED"                      # a changed heatmap shows up
    assert diffs[".x.probability_fake"] == "within tolerance"
    assert diffs[".x.explainability.model_consistency.std_probability"] == "random by design"
    assert not any("timestamp_utc" in path for path in diffs)                       # volatile, ignored
    b["x"]["probability_fake"] = 0.6
    assert dict((p, v) for p, _, _, v in snapshot.value_diffs(a, b))[".x.probability_fake"] == "CHANGED"


def test_monte_carlo_readings_are_never_counted_as_changes():
    assert snapshot.compare_values("consistency_std", 0.0478, 0.0961) == "random by design"
    assert snapshot.compare_values("probability_fake", 0.89, 0.90) == "CHANGED"


def test_whatsapp_pairs_of_one_kind_do_not_crash_the_summary():
    # The team's most likely first contribution: phone photos and their WhatsApp
    # copies only -- no AI pairs, so no AI average and no AUC, but no crash.
    by_id = {"o1": {**_rec("real", 0.10), "id": "o1"}, "o2": {**_rec("real", 0.30), "id": "o2"}}
    copies = [{**_rec("real", 0.15), "purpose": "robustness", "variant_of": "o1"},
              {**_rec("real", 0.15), "purpose": "robustness", "variant_of": "o2"}]   # inconclusive -> real
    res = run_exam.robustness_section(by_id, copies)
    assert res["pairs"] == 2 and res["verdict_changed"] == 1
    assert res["ai_images_mean_p_ai"] is None and res["auc"] == {"originals": None, "social_copies": None}
