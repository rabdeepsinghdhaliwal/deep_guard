"""
Deep-Guard -- Content Protection registry (Phase 2, Idea 4).

The existing fingerprint (fingerprint.py) can tell you "is this the exact
same file" (SHA-256) or "does this still look the same after
recompression" (perceptual hash) -- but a perceptual hash is a GLOBAL
description of the whole frame, so cropping it breaks the comparison
almost immediately (independently verified: PhotoDNA-family hashes lose
their match after a crop of only ~2.5% of the image edge). That's the
gap this module closes: "is this a crop, edit, or partial reuse of
something already registered" -- the actual mechanism behind stopping
someone from cropping a signature off a registered painting and
reselling it as new.

How: SSCD (Self-Supervised Descriptor for Copy Detection -- Meta AI,
CVPR 2022) turns an image into a 512-dim vector such that images sharing
substantial content land close together in that vector space, even
across cropping, rotation, recolouring, or heavy compression -- while
genuinely different images land far apart. This is Meta's own pretrained
model, used exactly as EfficientNet-B0 is used elsewhere in this project:
as a frozen feature extractor, never trained by this project.

Registered vectors live in a FAISS index (Meta's own open-source nearest-
neighbour search library) persisted to disk, so "search the registry by
content" survives server restarts, the same way the signing key does.

VERIFIED before this file was written a single line at a time (see this
session's prototype_sscd.py): on this project's own demo artworks,
different paintings scored at most 0.132 cosine similarity to each
other; the SAME painting cropped 40% still scored 0.698 to its own
original -- a >5x separation. MATCH_THRESHOLD below sits between those
two measured numbers, not picked blind.
"""

import json
import threading
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Identical to model.py's constants on purpose -- SSCD's own README
# specifies standard ImageNet normalization, so this isn't a coincidence
# to keep in sync, just the same well-known numbers reappearing.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Per SSCD's README: resize the short edge to 288, no centre-crop --
# unlike Stage 1's classifier, this model expects to see the whole
# frame, since cropping it here (rather than letting a REGISTERED
# image's crop be what we're testing robustness against) would throw
# away exactly the content the registry needs to compare.
PREPROCESS = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# Cosine similarity a candidate needs to clear an existing registration
# to count as "the same content". Measured empirically (see module
# docstring): different paintings topped out at 0.132; the same
# painting survived a 40% crop at 0.698. 0.4 sits with wide margin on
# both sides of that gap, not at either extreme.
MATCH_THRESHOLD = 0.4

EMBEDDING_DIM = 512


def load_sscd_model(model_path: Path) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path))
    model.eval()
    return model


def compute_content_embedding(model: torch.jit.ScriptModule, image: Image.Image) -> np.ndarray:
    """Returns a unit-length (L2-normalized) 512-dim vector, so that
    cosine similarity between two embeddings is just their dot product
    -- what ContentRegistry.search below actually computes via FAISS's
    inner-product index."""
    batch = PREPROCESS(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vector = model(batch)[0, :].numpy().astype("float32")
    norm = np.linalg.norm(vector)
    if norm > 1e-8:
        vector = vector / norm
    return vector


class ContentRegistry:
    """
    Wraps a FAISS IndexFlatIP (exact, brute-force inner-product search --
    entirely fast enough at the scale of a demo registry; an approximate
    index only starts to matter at millions of entries) plus a parallel
    JSON file of metadata, both persisted to disk so registrations
    survive a server restart, the same way the signing key does.
    """

    def __init__(self, index_path: Path, metadata_path: Path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.metadata: list[dict] = []
        # Found in round-1 review: register() is a read-modify-write
        # across two data structures (index.add + metadata.append) with
        # no atomicity between them. It happened to be safe only because
        # main.py's endpoint has no `await` between reading the upload
        # and calling register() -- Python's single-threaded event loop
        # can't interleave two requests without a yield point in between.
        # That's an implicit, fragile invariant (a future `await` added
        # anywhere in that path would silently reintroduce the race, and
        # nothing would flag it), not a real guarantee -- hence an actual
        # lock, cheap here since register() is fast (no model inference
        # happens inside it, only index/metadata bookkeeping).
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def is_registered(self, title: str) -> bool:
        """Used only to make demo-artwork seeding idempotent across
        restarts -- not a general duplicate-detection feature."""
        return any(entry["title"] == title for entry in self.metadata)

    def register(self, embedding: np.ndarray, record: dict) -> int:
        """record: the signed fingerprint record (from fingerprint.py) plus
        whatever display metadata (title, etc.) the caller wants stored.
        Returns the new entry's position in the index."""
        with self._lock:
            self.index.add(embedding.reshape(1, -1))
            self.metadata.append(record)
            self.save()
            return len(self.metadata) - 1

    def search(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Returns up to k matches with similarity >= MATCH_THRESHOLD,
        strongest first. An empty registry or no qualifying match both
        just return an empty list -- there is no ambiguous "close but
        not sure" state this exposes to the caller."""
        with self._lock:  # same lock as register() -- FAISS doesn't guarantee add()/search() are safe concurrently
            if self.index.ntotal == 0:
                return []
            k = min(k, self.index.ntotal)
            similarities, indices = self.index.search(embedding.reshape(1, -1), k)
            metadata_snapshot = self.metadata

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < 0 or similarity < MATCH_THRESHOLD:
                continue
            results.append({
                "similarity": round(float(similarity), 4),
                **metadata_snapshot[idx],
            })
        return results
