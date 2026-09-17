"""
Integration tests for GET /api/bias-map and GET /bias-map. Reads the
already-committed bias_map/results.json -- no computation happens on
this request, so these are fast even though they go through `client`.
"""


def test_bias_map_page_loads(client):
    response = client.get("/bias-map")
    assert response.status_code == 200


def test_bias_map_api_reports_generated(client):
    response = client.get("/api/bias-map")
    assert response.status_code == 200
    data = response.json()

    assert data["generated"] is True
    assert data["total_images"] == 8
    for key in ("by_subject", "by_style", "by_generator", "by_combination"):
        assert key in data

    # The committed results.json is an honest placeholder (see its own
    # scope_warning) -- pin that the warning is still there rather than
    # silently dropped, since it's load-bearing for not misrepresenting
    # this data as trustworthy.
    assert "scope_warning" in data
    assert "8 images" in data["scope_warning"]
