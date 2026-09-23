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
import math
import threading
from pathlib import Path
from typing import Optional

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


def compute_content_features(model: torch.jit.ScriptModule, image: Image.Image) -> tuple:
    """The same unit-length vector as compute_content_embedding, plus the
    spatial feature map it was pooled from, in one forward pass.

    SSCD is two stages (visible inside the TorchScript file): a ResNet-50
    `backbone` that turns the image into a grid of 2048-number cells --
    one per ~32x32 pixel block of the 288-short-edge input, so 9 x 14 for
    the Mona Lisa -- and `embeddings`, which pools that grid into one
    vector (GeM pooling), projects it to 512 numbers and L2-normalises it.
    Keeping the grid is what lets similarity_shares() below say which
    parts of an image the match came from."""
    batch = PREPROCESS(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feature_map = model.backbone(batch)
        vector = model.embeddings(feature_map)[0, :].numpy().astype("float32")
    norm = np.linalg.norm(vector)
    if norm > 1e-8:
        vector = vector / norm
    return vector, feature_map


# GeM pooling constants, read straight out of the released model's
# TorchScript code (`embeddings.0`): clamp(x, 1e-6) ** 3, mean over cells,
# then ** (1/3).
_GEM_EPS = 1e-6
_GEM_P = 3.0

# 2**MAX_SHAPLEY_PLAYERS coalitions are evaluated exactly. 3x3 areas plus
# "not from the original" is 10.
MAX_SHAPLEY_PLAYERS = 12


def similarity_shares(model: torch.jit.ScriptModule, feature_map: torch.Tensor,
                      reference_vector: np.ndarray, owners: list) -> tuple:
    """
    Splits the registry's similarity score between an image and one stored
    vector into exact shares, one per group of feature-map cells
    ("players" -- e.g. "the cells that show the original's face").

    Fair credit is computed with Shapley values, the same idea as the SHAP
    panel on the detector (shap_explain.py): a player's share is its
    average contribution over every order the players could be added in.
    Here it is EXACT, not sampled. GeM pooling is a mean over cells, so the
    pooled vector of any subset of cells follows from per-player sums of
    cubed features -- every coalition's similarity is one small matrix
    product through the model's own projection layers, no extra network
    passes. The shares add up to the full similarity by construction (the
    Shapley "efficiency" property), and the full coalition reproduces the
    registry's own score, which tests pin.

    owners: one hashable label per cell, in the feature map's row-major
    order (feature_map.flatten order). Returns ({owner: share}, total).
    """
    players = list(dict.fromkeys(owners))
    n = len(players)
    if n == 0:
        return {}, 0.0
    if n > MAX_SHAPLEY_PLAYERS:
        raise ValueError(f"{n} players is too many for an exact split (limit {MAX_SHAPLEY_PLAYERS}).")

    cells = feature_map[0].flatten(1).clamp_min(_GEM_EPS).pow(_GEM_P)  # [C, cells]
    owner_of = np.array([players.index(o) for o in owners])
    group_sums = torch.stack([cells[:, owner_of == j].sum(dim=1) for j in range(n)])  # [n, C]
    group_counts = torch.tensor([(owner_of == j).sum() for j in range(n)], dtype=torch.float32)

    masks = torch.tensor([[(m >> j) & 1 for j in range(n)] for m in range(1, 2 ** n)], dtype=torch.float32)
    with torch.no_grad():
        pooled = ((masks @ group_sums) / (masks @ group_counts).unsqueeze(1)).pow(1.0 / _GEM_P)
        projected = getattr(model.embeddings, "1")(pooled)
        normalised = getattr(model.embeddings, "2")(projected)
    values = np.zeros(2 ** n)
    values[1:] = normalised.double().numpy() @ reference_vector.astype(np.float64)

    shares = np.zeros(n)
    weights = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    for mask in range(2 ** n):
        size = bin(mask).count("1")
        for j in range(n):
            if not mask & (1 << j):
                shares[j] += weights[size] * (values[mask | (1 << j)] - values[mask])
    return {p: float(s) for p, s in zip(players, shares)}, float(values[-1])


def _record_key(record: dict) -> str:
    """Identifies one registration, so a features file left over from a
    registry that was deleted and rebuilt can't be attached to the wrong
    entry (entry ids restart from 0)."""
    return f"{record.get('sha256_merkle_root', '')}|{record.get('timestamp_utc', '')}"


class ContentRegistry:
    """
    Wraps a FAISS IndexFlatIP (exact, brute-force inner-product search --
    entirely fast enough at the scale of a demo registry; an approximate
    index only starts to matter at millions of entries) plus a parallel
    JSON file of metadata, both persisted to disk so registrations
    survive a server restart, the same way the signing key does.
    """

    def __init__(self, index_path: Path, metadata_path: Path, features_dir: Optional[Path] = None):
        self.index_path = index_path
        self.metadata_path = metadata_path
        # One .npz per entry holding what registry_explain.py needs to
        # explain a match later (distinctive points, area names, colour
        # grid) -- never pixels. Optional: without it the registry matches
        # exactly as before and explanations report "unavailable".
        self.features_dir = features_dir
        if features_dir is not None:
            features_dir.mkdir(parents=True, exist_ok=True)
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

    def entry_ids_titled(self, title: str) -> list[int]:
        return [i for i, entry in enumerate(self.metadata) if entry["title"] == title]

    def register(self, embedding: np.ndarray, record: dict, features: Optional[dict] = None) -> int:
        """record: the signed fingerprint record (from fingerprint.py) plus
        whatever display metadata (title, etc.) the caller wants stored.
        features: optional arrays for later explanations (see
        registry_explain.storable), written before the entry itself so a
        failed write leaves nothing half-registered.
        Returns the new entry's position in the index."""
        with self._lock:
            entry_id = len(self.metadata)
            if features is not None and self.features_dir is not None:
                self._write_features(entry_id, record, features)
            self.index.add(embedding.reshape(1, -1))
            self.metadata.append(record)
            self.save()
            return entry_id

    def _features_path(self, entry_id: int) -> Path:
        return self.features_dir / f"{entry_id:06d}.npz"

    def _write_features(self, entry_id: int, record: dict, features: dict) -> None:
        np.savez_compressed(self._features_path(entry_id), record_key=np.array(_record_key(record)), **features)

    def save_features(self, entry_id: int, features: dict) -> None:
        """Attaches features to an entry registered without them (used to
        backfill the demo artworks registered before explanations existed)."""
        if self.features_dir is None:
            return
        with self._lock:
            self._write_features(entry_id, self.metadata[entry_id], features)

    def load_features(self, entry_id: int) -> Optional[dict]:
        """The entry's stored feature arrays, or None if it has none -- or
        if the file on disk belongs to a different registration."""
        if self.features_dir is None or not 0 <= entry_id < len(self.metadata):
            return None
        path = self._features_path(entry_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        if str(arrays.pop("record_key", "")) != _record_key(self.metadata[entry_id]):
            return None
        return arrays

    def stored_vector(self, entry_id: int) -> np.ndarray:
        """The unit-length SSCD vector this entry was registered with."""
        with self._lock:
            return self.index.reconstruct(int(entry_id))

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
                "entry_id": int(idx),
                **metadata_snapshot[idx],
            })
        return results
