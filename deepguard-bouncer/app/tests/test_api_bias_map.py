"""
Integration tests for GET /api/bias-map and GET /bias-map. Reads the
already-committed bias_map/results.json -- no computation happens on
this request, so these are fast even though they go through `client`.
That file is written by evaluation/run_exam.py --write-bias-map from the
exam set (evaluation/exam_set_manifest.csv).
"""


def test_bias_map_page_loads(client):
    response = client.get("/bias-map")
    assert response.status_code == 200


def test_bias_map_api_reports_generated(client):
    response = client.get("/api/bias-map")
    assert response.status_code == 200
    data = response.json()

    assert data["generated"] is True
    # The exam set, not the old 8-image placeholder.
    assert data["total_images"] >= 300
    assert "exam set" in data["scope_note"]
    for key in ("caught", "false_alarm", "missed", "inconclusive", "auc"):
        assert key in data["headline"]
    # Kept for older copies of the page.
    for key in ("by_subject", "by_style", "by_generator", "by_combination"):
        assert key in data


def test_bias_map_groups_are_consistent(client):
    data = client.get("/api/bias-map").json()
    assert data["sections"]
    for section in data["sections"]:
        assert section["title"] and section["groups"]
        for name, group in section["groups"].items():
            assert group["sample_count"] > 0
            assert 0.0 <= group["accuracy"] <= 1.0
            # Right + wrong + unsure accounts for every image in the group.
            assert abs(group["accuracy"] + group["wrong"] + group["inconclusive"] - 1.0) < 1e-3
            # Small groups must be flagged, never shown with the same authority.
            assert group["trustworthy"] == (group["sample_count"] >= data["min_trustworthy_group_size"])
