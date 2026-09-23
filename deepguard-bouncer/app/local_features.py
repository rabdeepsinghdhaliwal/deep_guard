"""
Deep-Guard -- distinctive points, for explaining Content Registry matches.

The registry decides "is this a copy?" with SSCD (content_registry.py): one
512-number summary per image, compared as a whole. That is fast and robust,
but it can't say WHICH parts of two images are the same, and a summary with
no parts can't be read as a list of shared features. This module adds the
classical, point-by-point evidence a person can inspect -- the same step a
fingerprint examiner performs when they mark matching ridge points on two
prints instead of just saying "they look alike":

  1. Distinctive points: SIFT (Lowe, "Distinctive Image Features from
     Scale-Invariant Keypoints", IJCV 2004). Corners and blobs whose
     128-number descriptions survive resizing, rotation, recolouring and
     compression. Compared as RootSIFT (Arandjelovic & Zisserman, CVPR
     2012), a small change that makes the comparison noticeably more
     reliable.
  2. Matching: each point in the upload is paired with its closest point
     in the original, kept only if it's clearly closer than the runner-up
     (Lowe's ratio test, 0.8 as in his paper).
  3. Verification: most accidental pairs disagree about where they sit.
     RANSAC finds the single rotation + scale + shift that the most pairs
     agree on; only those pairs ("inliers") count as shared points. This is
     the spatial-verification step large image-search systems have used
     since Philbin et al. (CVPR 2007), and local verification is what the
     top system of Meta's 2021 Image Similarity Challenge (matching track)
     added on top of a global embedding like ours.
  4. The fitted transform itself is evidence of what was done to the copy:
     how much of the original it shows and from where, rotation, mirroring,
     size, added borders -- and, comparing colours cell by cell once the
     two are aligned, whether colour was removed, changed or brightened.

MEASURED (registry_explanations_test/run_measurements.py at the repo
root): all 24 real edits of the demo paintings -- crops down to 25%,
15 and 90 degree rotations, a mirror, black and white, sepia, brightening,
blur, JPEG q=10, a text overlay, an added border -- were confirmed by 61
to 1,504 shared points with every recovered fact correct, while 594
pairs of unrelated images shared at most 9 points by chance.

What a registration keeps from this module: point positions, sizes and
descriptors, plus an 8x8 grid of average colours -- never pixels. Note
that research has reconstructed rough images from SIFT descriptors
(Weinzaepfel et al., CVPR 2011), so stored features are treated as
sensitive data, not as harmless numbers.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Every image is examined at this size (long side, never upscaled): point
# scales become comparable between an original and its copy, and a huge
# upload can't make SIFT slow.
FRAME_LONG_SIDE = 1024

# Strongest points kept. A registration stores up to MAX_STORED_POINTS
# (~200 KB with descriptors); an upload being checked gets more, so a small
# crop of a registered image still has plenty to match.
MAX_STORED_POINTS = 1500
MAX_QUERY_POINTS = 2500

LOWE_RATIO = 0.8

# A pair counts as agreeing with the fitted placement if it lands within
# 1% of the original's long side (~10 px on a 1024 frame).
RANSAC_TOLERANCE = 0.01

# Colour is compared on an 8x8 grid of average colours: enough to tell
# "black and white" from "tinted" from "unchanged", far too coarse to be a
# picture of the original.
COLOUR_GRID = 8

# A mirrored fit has to beat the normal one clearly before a copy is called
# mirrored -- a left-right symmetric image matches well both ways.
MIRROR_MARGIN = 1.2

AREA_POSITIONS = [
    "top left", "top centre", "top right",
    "middle left", "centre", "middle right",
    "bottom left", "bottom centre", "bottom right",
]


@dataclass
class LocalFeatures:
    frame_size: tuple       # (w, h) of the examined frame
    source_size: tuple      # (w, h) of the image as uploaded
    points: np.ndarray      # [N, 2] float32, frame pixels, strongest first
    sizes: np.ndarray       # [N] float32, keypoint diameter in frame pixels
    descriptors: np.ndarray  # [N, 128] uint8, raw SIFT
    colour_grid: np.ndarray  # [8, 8, 3] uint8, OpenCV 8-bit CIELAB

    @property
    def count(self) -> int:
        return int(len(self.points))

    def to_arrays(self) -> dict:
        return {
            "frame_size": np.array(self.frame_size, np.int32),
            "source_size": np.array(self.source_size, np.int32),
            "points": self.points.astype(np.float32),
            "sizes": self.sizes.astype(np.float32),
            "descriptors": self.descriptors.astype(np.uint8),
            "colour_grid": self.colour_grid.astype(np.uint8),
        }

    @classmethod
    def from_arrays(cls, data) -> "LocalFeatures":
        return cls(
            frame_size=tuple(int(v) for v in data["frame_size"]),
            source_size=tuple(int(v) for v in data["source_size"]),
            points=np.asarray(data["points"], np.float32),
            sizes=np.asarray(data["sizes"], np.float32),
            descriptors=np.asarray(data["descriptors"], np.uint8),
            colour_grid=np.asarray(data["colour_grid"], np.uint8),
        )


@dataclass
class Verification:
    query_idx: np.ndarray    # inlier pairs: index into the query's points...
    ref_idx: np.ndarray      # ...and the matching index into the reference's
    transform: Optional[np.ndarray]  # 2x3, query frame (mirrored if `mirrored`) -> reference frame
    mirrored: bool
    candidates: int          # pairs that passed the ratio test, before RANSAC

    @property
    def shared_points(self) -> int:
        return int(len(self.query_idx))


def examination_frame(image: Image.Image) -> np.ndarray:
    """The RGB frame everything in this module works on: the upload,
    shrunk (never enlarged) so its long side is at most FRAME_LONG_SIDE."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = min(1.0, FRAME_LONG_SIDE / max(w, h))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def colour_grid(frame_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    return cv2.resize(lab, (COLOUR_GRID, COLOUR_GRID), interpolation=cv2.INTER_AREA)


def extract(frame_rgb: np.ndarray, source_size: tuple, max_points: int) -> LocalFeatures:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=max_points).detectAndCompute(gray, None)
    h, w = gray.shape
    if descriptors is None or not keypoints:
        points, sizes, descriptors = np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros((0, 128), np.uint8)
    else:
        # nfeatures keeps the strongest by response but can return a few
        # extra on ties; sort and cap so "strongest first" is guaranteed.
        order = np.argsort([-k.response for k in keypoints], kind="stable")[:max_points]
        points = np.float32([keypoints[i].pt for i in order])
        sizes = np.float32([keypoints[i].size for i in order])
        descriptors = np.clip(np.rint(descriptors[order]), 0, 255).astype(np.uint8)
    return LocalFeatures(
        frame_size=(w, h), source_size=tuple(source_size), points=points, sizes=sizes,
        descriptors=descriptors, colour_grid=colour_grid(frame_rgb),
    )


def mirror(features: LocalFeatures, frame_rgb: np.ndarray, max_points: int) -> LocalFeatures:
    """Features of the left-right mirrored frame. SIFT descriptors aren't
    mirror-invariant, so a mirrored copy is matched by describing the
    upload flipped back, not by comparing the descriptors as they are."""
    return extract(np.ascontiguousarray(frame_rgb[:, ::-1]), features.source_size, max_points)


def _rootsift(descriptors: np.ndarray) -> np.ndarray:
    d = descriptors.astype(np.float32)
    d /= d.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(d)


def _ratio_pairs(query: LocalFeatures, reference: LocalFeatures) -> list:
    if query.count < 2 or reference.count < 2:
        return []
    pairs = []
    for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(_rootsift(query.descriptors), _rootsift(reference.descriptors), k=2):
        if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance:
            pairs.append((m[0].queryIdx, m[0].trainIdx))
    return pairs


def _fit(query: LocalFeatures, reference: LocalFeatures, pairs: list) -> tuple:
    empty = np.zeros(0, np.int64)
    if len(pairs) < 4:
        return None, empty, empty
    q_idx = np.array([p[0] for p in pairs])
    r_idx = np.array([p[1] for p in pairs])
    transform, mask = cv2.estimateAffinePartial2D(
        query.points[q_idx], reference.points[r_idx], method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_TOLERANCE * max(reference.frame_size),
        maxIters=5000, confidence=0.995,
    )
    if transform is None:
        return None, empty, empty
    scale = math.hypot(transform[0, 0], transform[1, 0])
    if not 0.05 < scale < 20:  # a "fit" that squashes or blows up the image 20x is noise
        return None, empty, empty
    keep = mask.ravel().astype(bool)
    return transform, q_idx[keep], r_idx[keep]


def verify(query: LocalFeatures, reference: LocalFeatures, query_mirrored: Optional[LocalFeatures] = None) -> Verification:
    """Shared, geometrically consistent points between an upload and one
    registered original, trying the upload mirrored too when given."""
    pairs = _ratio_pairs(query, reference)
    transform, q_idx, r_idx = _fit(query, reference, pairs)
    best = Verification(q_idx, r_idx, transform, False, len(pairs))
    if query_mirrored is not None:
        m_pairs = _ratio_pairs(query_mirrored, reference)
        m_transform, m_q, m_r = _fit(query_mirrored, reference, m_pairs)
        if len(m_q) > MIRROR_MARGIN * best.shared_points and len(m_q) - best.shared_points >= 5:
            best = Verification(m_q, m_r, m_transform, True, len(m_pairs))
    return best


def area_index(x_norm: float, y_norm: float) -> Optional[int]:
    """Which of the 3x3 areas a point (normalised 0-1 coordinates) is in,
    or None if it lies outside the frame."""
    if not (0.0 <= x_norm < 1.0 and 0.0 <= y_norm < 1.0):
        return None
    return int(y_norm * 3) * 3 + int(x_norm * 3)


def describe_geometry(verification: Verification, query: LocalFeatures, reference: LocalFeatures) -> dict:
    """Turns the fitted placement into plain facts about the copy."""
    transform = verification.transform
    a, b = float(transform[0, 0]), float(transform[1, 0])
    scale = math.hypot(a, b)
    # Image y points down, so a positive angle here means the upload is
    # turned anticlockwise relative to the original (PIL's rotate(+15)
    # comes out as +15.0).
    rotation = math.degrees(math.atan2(b, a))
    if abs(rotation) < 1.5:
        rotation = 0.0

    # How big the content appears in the upload file compared with the
    # original file: frame scales undo the examination-frame resizing.
    fq = query.frame_size[0] / max(query.source_size[0], 1)
    fr = reference.frame_size[0] / max(reference.source_size[0], 1)
    size_ratio = fr / (fq * scale)

    wq, hq = query.frame_size
    wr, hr = reference.frame_size
    corners = np.float32([[0, 0], [wq, 0], [wq, hq], [0, hq]])
    footprint = corners @ transform[:, :2].T.astype(np.float32) + transform[:, 2].astype(np.float32)
    frame = np.float32([[0, 0], [wr, 0], [wr, hr], [0, hr]])
    overlap, overlap_poly = cv2.intersectConvexConvex(footprint.astype(np.float32), frame)
    footprint_area = abs(float(cv2.contourArea(footprint.astype(np.float32))))
    shown = float(overlap) / (wr * hr)
    added = 1.0 - float(overlap) / footprint_area if footprint_area > 0 else 0.0

    if shown >= 0.85:
        position = "almost all of it"
    elif overlap_poly is not None and len(overlap_poly):
        cx, cy = overlap_poly.reshape(-1, 2).mean(axis=0)
        position = AREA_POSITIONS[area_index(min(cx / wr, 0.999), min(cy / hr, 0.999))]
    else:
        position = None

    return {
        "rotation_deg": round(rotation, 1),
        "mirrored": verification.mirrored,
        "size_ratio": round(size_ratio, 2),
        "original_area_shown": round(min(shown, 1.0), 3),
        "shown_position": position,
        "added_area": round(max(added, 0.0), 3),
        "footprint": [[round(float(x) / wr, 4), round(float(y) / hr, 4)] for x, y in footprint],
    }


def _lab(grid_u8: np.ndarray) -> tuple:
    g = grid_u8.astype(np.float32)
    return g[..., 0] * (100.0 / 255.0), g[..., 1] - 128.0, g[..., 2] - 128.0


def compare_colour(query_frame_rgb: np.ndarray, verification: Verification, reference: LocalFeatures) -> dict:
    """Aligns the upload onto the original with the fitted placement, then
    compares average colours cell by cell on the original's 8x8 grid --
    so a crop is compared with the part of the original it came from, not
    with the whole painting."""
    frame = query_frame_rgb[:, ::-1] if verification.mirrored else query_frame_rgb
    frame = np.ascontiguousarray(frame)
    wr, hr = reference.frame_size
    warped = cv2.warpAffine(frame, verification.transform, (wr, hr), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), verification.transform, (wr, hr), flags=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)

    ref_l, ref_a, ref_b = _lab(reference.colour_grid)
    rows = []
    for r in range(COLOUR_GRID):
        y0, y1 = r * hr // COLOUR_GRID, (r + 1) * hr // COLOUR_GRID
        for c in range(COLOUR_GRID):
            x0, x1 = c * wr // COLOUR_GRID, (c + 1) * wr // COLOUR_GRID
            cell_valid = valid[y0:y1, x0:x1] > 0
            if cell_valid.size == 0 or cell_valid.mean() < 0.6:
                continue
            q = lab[y0:y1, x0:x1][cell_valid].mean(axis=0)
            rows.append((q[0] * 100.0 / 255.0, q[1] - 128.0, q[2] - 128.0, ref_l[r, c], ref_a[r, c], ref_b[r, c]))
    if len(rows) < 2:
        return {"change": "unknown", "brightness": "unknown", "cells_compared": len(rows)}

    t = np.array(rows, np.float32)
    ql, qa, qb, ol, oa, ob = t.T
    q_chroma, o_chroma = np.hypot(qa, qb), np.hypot(oa, ob)
    chroma_ratio = float(q_chroma.mean() / max(o_chroma.mean(), 1e-6))
    both = (q_chroma >= 8) & (o_chroma >= 8)
    hue_shift = None
    if both.sum() >= 3:
        diff = np.degrees(np.arctan2(qb, qa) - np.arctan2(ob, oa))[both]
        hue_shift = float(np.abs((diff + 180.0) % 360.0 - 180.0).mean())
    lightness_shift = float((ql - ol).mean())
    # Average distance between matching cells in the colour plane (a*, b*),
    # brightness ignored: catches a tint that moves every colour a little
    # even when the average hue barely turns (sepia on an already-brown
    # painting), which a hue-angle test alone misses.
    colour_shift = float(np.hypot(qa - oa, qb - ob).mean())

    # Thresholds measured on the 18 Mona Lisa edits and 7 Great Wave edits:
    # unchanged copies (crops, rotations, JPEG q=10...) shifted colour by
    # 0.3-2.4 and hue by at most 7.8 degrees; sepia shifted 15.5-17.4;
    # brightening also moves colour (9.2-10.3) but always together with a
    # large lightness change, so it is reported as brightness, not a tint.
    brightness_changed = abs(lightness_shift) >= 8
    if o_chroma.mean() < 6:
        change = "kept"  # the original has almost no colour to lose
    elif chroma_ratio < 0.3:
        change = "removed"
    elif (hue_shift is not None and hue_shift >= 25) or (colour_shift >= 6 and not brightness_changed):
        change = "recoloured"
    elif chroma_ratio < 0.65 and not brightness_changed:
        change = "faded"
    elif chroma_ratio > 1.5 and not brightness_changed:
        change = "intensified"
    else:
        change = "kept"
    brightness = "brighter" if lightness_shift >= 8 else "darker" if lightness_shift <= -8 else "same"
    return {
        "change": change,
        "brightness": brightness,
        "chroma_ratio": round(chroma_ratio, 2),
        "hue_shift_deg": round(hue_shift, 1) if hue_shift is not None else None,
        "lightness_shift": round(lightness_shift, 1),
        "colour_shift": round(colour_shift, 1),
        "cells_compared": len(rows),
    }


def palette(frame_rgb: np.ndarray, k: int = 5) -> list:
    """The image's k dominant colours with their share of the picture
    (k-means in CIELAB, so 'close' means close to the eye)."""
    small = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)  # k-means starts from random centres; fixed so the same image always gets the same palette
    _, labels, centres = cv2.kmeans(
        lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=k)
    rgb = cv2.cvtColor(np.clip(centres, 0, 255).astype(np.uint8).reshape(1, k, 3), cv2.COLOR_LAB2RGB).reshape(k, 3)
    merged = {}
    for (r, g, b), n in zip(rgb, counts):
        key = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        merged[key] = merged.get(key, 0) + int(n)
    total = max(sum(merged.values()), 1)
    return [{"hex": h, "share": round(n / total, 3)} for h, n in sorted(merged.items(), key=lambda kv: -kv[1]) if n]
