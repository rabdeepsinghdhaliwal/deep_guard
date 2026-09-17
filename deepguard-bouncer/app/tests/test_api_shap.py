"""
Integration test for POST /api/explain/shap-regions. Slow relative to
the rest of the suite -- SHAP re-runs inference on ~200 masked
variants of the image on CPU -- so kept to a single representative
image rather than parametrized across all 8.
"""


def test_shap_regions_returns_well_formed_contributions(client, test_images):
    response = client.post(
        "/api/explain/shap-regions",
        files={"file": ("ai1.png", test_images["ai1.png"], "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["regions"], list)
    assert len(data["regions"]) > 0
    assert data["segments_total"] > 0

    for region in data["regions"]:
        assert "contribution" in region
        assert "thumbnail_png_base64" in region
        assert isinstance(region["contribution"], float)


def test_shap_regions_rejects_video_extension(client):
    response = client.post(
        "/api/explain/shap-regions",
        files={"file": ("clip.mp4", b"not a real video", "video/mp4")},
    )
    assert response.status_code == 400
