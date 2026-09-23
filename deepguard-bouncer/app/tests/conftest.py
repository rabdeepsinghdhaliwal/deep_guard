"""
Shared fixtures for the Deep-Guard test suite.

Run from the app/ directory (`cd app && pytest`) -- same working
directory uvicorn already runs from, so `import main` etc. work
exactly like they do in production, no path hacks.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

# test_images/ lives at the repo root: deep_guard/test_images/, three
# levels up from deep_guard/deepguard-bouncer/app/tests/conftest.py.
TEST_IMAGES_DIR = Path(__file__).resolve().parents[3] / "test_images"

TEST_IMAGE_FILES = [
    "ai1.png", "ai2.png", "ai3.png", "ai4.png",
    "real1.png", "real2.jpg", "real3.jpg", "real4.jpg",
]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """
    A single TestClient shared by the whole session, so the ~130MB of
    real model weights (base detector + attribution + SSCD) load once,
    not once per test.

    The Content Registry's persisted files (index, metadata, features)
    are redirected to a session-scoped tmp directory BEFORE the
    TestClient is created --
    TestClient triggers main.py's lifespan() startup (which seeds the
    registry from demo_artworks/) on __enter__, so this has to happen
    first or it seeds straight into the real models/content_registry.*
    files, the exact test-pollution problem manually cleaned up
    several times earlier this project.
    """
    registry_dir = tmp_path_factory.mktemp("registry")
    main.CONTENT_REGISTRY_INDEX_PATH = registry_dir / "test_registry.index"
    main.CONTENT_REGISTRY_METADATA_PATH = registry_dir / "test_registry_metadata.json"
    main.CONTENT_REGISTRY_FEATURES_DIR = registry_dir / "features"

    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def test_images():
    """dict[filename] -> raw bytes, for the 8 real labeled demo images."""
    return {name: (TEST_IMAGES_DIR / name).read_bytes() for name in TEST_IMAGE_FILES}
