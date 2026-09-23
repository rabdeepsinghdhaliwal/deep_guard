"""
Deep-Guard -- what the Content Registry can say about an image beyond a
score: its distinct features when it is uploaded, and, for a match, what
the two images have in common and how much each common part counted.

Three independent pieces, each measured before it was used:

  * local_features.py -- distinctive points (SIFT) and the verified
    placement of the upload inside the original: which exact points are
    shared, and what was done to the copy (crop, rotation, mirror, size,
    border, colour).
  * concept_labels.py -- names for the 3x3 areas of the ORIGINAL, taken
    once when it is registered ("face", "hands", "landscape"...). A name
    is a model's guess and is always shown as one.
  * content_registry.similarity_shares -- the registry's own SSCD score,
    split exactly among those named things with Shapley values. The
    shares add back up to the reported similarity, so "what was
    responsible for the detection" is answered by the network that made
    the detection, not by a different method.

Only the SSCD score decides whether something is a match
(content_registry.MATCH_THRESHOLD). Everything here explains a decision
that has already been made; nothing here can make or veto one.
"""

import base64
import io
import json
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import content_registry
import local_features as lf

FEATURE_VERSION = 1

# Each 3x3 area is named from a crop reaching 10% of the image further on
# every side. Tuned on the 3 demo paintings + 8 test photos: without the
# context the Mona Lisa's face came out "child" and Starry Night's village
# "windmill"; with it, "face" and "village".
AREA_CONTEXT = 0.10

# Shared points (RANSAC inliers) before the fitted placement is trusted.
# MEASURED (registry_explanations_test/run_measurements.py): 594 pairs of
# unrelated images -- the demo paintings and test photos against each
# other and against 44 MidJourney / DALL-E 3 images -- shared at most 9
# points by chance (mean 1.4); all 24 real edits of registered paintings
# shared 61 to 1,504. "Weak" starts above anything chance produced;
# "confirmed" sits with a wide margin on both sides of that gap.
WEAK_POINTS = 10
CONFIRMED_POINTS = 20
STRONG_POINTS = 50

# A runner-up name is shown only when the namer genuinely hesitated.
RUNNER_UP_MIN = 0.15

# The vocabulary has "background" names so an empty area isn't forced onto
# the nearest object. But when the namer only weakly prefers a background
# over a real thing, the thing is both more useful and usually right --
# measured on the test photo of a face: its centre came out "blurry
# background" (0.36) with "face" as the runner-up. A confident background
# (the Mona Lisa's dark lower right, 0.85) is kept.
BACKGROUND_NAMES = {"plain background", "dark background", "white background", "blurry background"}
WEAK_BACKGROUND = 0.5

PREVIEW_LONG_SIDE = 560
DRAWN_POINTS = 300          # points drawn on a distinct-features figure
FIGURE_REFERENCE_POINTS = 500
FIGURE_LINES = 90


@dataclass
class Upload:
    """One uploaded image, examined once and reused for every match."""
    image: Image.Image
    frame: np.ndarray
    features: lf.LocalFeatures
    areas: list
    things: list
    summary: dict
    max_points: int
    _mirrored: Optional[lf.LocalFeatures] = None

    def mirrored(self) -> lf.LocalFeatures:
        if self._mirrored is None:
            self._mirrored = lf.mirror(self.features, self.frame, self.max_points)
        return self._mirrored


def _area_boxes(w: int, h: int, context: float) -> list:
    boxes = []
    for r in range(3):
        for c in range(3):
            dx, dy = context * w, context * h
            boxes.append((
                max(0, int(c * w / 3 - dx)), max(0, int(r * h / 3 - dy)),
                min(w, int(round((c + 1) * w / 3 + dx))), min(h, int(round((r + 1) * h / 3 + dy))),
            ))
    return boxes


def name_areas(frame: np.ndarray, features: lf.LocalFeatures, labeler) -> list:
    """The 9 areas of an image: where they are, how many distinctive
    points each holds, and (with a namer) what each looks like."""
    h, w = frame.shape[:2]
    counts = Counter(lf.area_index(x / w, y / h) for x, y in features.points)
    guesses = None
    if labeler is not None:
        pil = Image.fromarray(frame)
        guesses = labeler.name_things([pil.crop(box) for box in _area_boxes(w, h, AREA_CONTEXT)], top_k=2)
    areas = []
    for i, position in enumerate(lf.AREA_POSITIONS):
        area = {"position": position, "points": int(counts.get(i, 0)), "name": None, "confidence": None, "runner_up": None}
        if guesses is not None:
            (name, p), (alt, q) = guesses[i][0], guesses[i][1]
            if name in BACKGROUND_NAMES and p < WEAK_BACKGROUND and q >= RUNNER_UP_MIN and alt not in BACKGROUND_NAMES:
                (name, p), (alt, q) = (alt, q), (name, p)
            area.update(name=name, confidence=round(p, 3), runner_up=alt if q >= RUNNER_UP_MIN else None)
        areas.append(area)
    return areas


def group_things(areas: list) -> list:
    """Areas given the same name become one 'thing' ("landscape" = top
    left + top right + middle left). Without a namer, every area is its
    own thing, named by position."""
    groups = {}
    for i, area in enumerate(areas):
        key = area["name"] or f"{area['position']} area"
        group = groups.setdefault(key, {"name": key, "named": area["name"] is not None, "areas": [],
                                        "points": 0, "confidences": []})
        group["areas"].append(i)
        group["points"] += area["points"]
        if area["confidence"] is not None:
            group["confidences"].append(area["confidence"])
    return [{
        "name": g["name"],
        "named": g["named"],
        "areas": g["areas"],
        "positions": [lf.AREA_POSITIONS[i] for i in g["areas"]],
        "points": g["points"],
        "confidence": round(float(np.mean(g["confidences"])), 3) if g["confidences"] else None,
    } for g in groups.values()]


def _preview_base64(frame: np.ndarray) -> str:
    """The upload as the server examined it (orientation included), for the
    page to draw on -- returned with the response, never stored."""
    preview = Image.fromarray(frame)
    preview.thumbnail((PREVIEW_LONG_SIDE, PREVIEW_LONG_SIDE))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def describe(image: Image.Image, labeler, max_points: int) -> Upload:
    """Everything the registry notices about an image: its distinctive
    points, what each of its 9 areas looks like, its colours and medium."""
    frame = lf.examination_frame(image)
    features = lf.extract(frame, image.size, max_points)
    areas = name_areas(frame, features, labeler)
    things = group_things(areas)
    medium = None
    if labeler is not None:
        name, p = labeler.name_medium(Image.fromarray(frame))
        medium = {"name": name, "confidence": round(p, 3)}
    h, w = frame.shape[:2]
    long_side = max(w, h)
    drawn = [[round(float(x) / w, 4), round(float(y) / h, 4), round(float(s) / 2 / long_side, 4)]
             for (x, y), s in zip(features.points[:DRAWN_POINTS], features.sizes[:DRAWN_POINTS])]
    summary = {
        "size": list(image.size),
        "points": {"count": features.count, "drawn": drawn},
        "areas": areas,
        "things": sorted(things, key=lambda t: -t["points"]),
        "palette": lf.palette(frame),
        "medium": medium,
        "namer": labeler.model_card if labeler is not None else None,
        "preview_jpeg_base64": _preview_base64(frame),
    }
    return Upload(image=image, frame=frame, features=features, areas=areas, things=things,
                  summary=summary, max_points=max_points)


def storable(upload: Upload, labeler) -> dict:
    """What a registration keeps for explaining future matches: point
    positions and descriptors, an 8x8 colour grid, and the area names.
    No pixels."""
    meta = {
        "feature_version": FEATURE_VERSION,
        "areas": upload.areas,
        "palette": upload.summary["palette"],
        "medium": upload.summary["medium"],
        "namer": labeler.model_card if labeler is not None else None,
        "vocab_hash": labeler.vocab_hash if labeler is not None else None,
    }
    return {**upload.features.to_arrays(), "meta_json": np.array(json.dumps(meta))}


def stored_size_kb(arrays: dict) -> int:
    return int(round(sum(a.nbytes for a in arrays.values()) / 1024))


def stored_meta(arrays: dict) -> dict:
    return json.loads(str(arrays["meta_json"]))


def _cell_centres(feature_map) -> np.ndarray:
    """Centres of SSCD's feature-map cells as 0-1 coordinates of the upload,
    in the same row-major order similarity_shares expects."""
    rows, cols = feature_map.shape[2], feature_map.shape[3]
    return np.array([((c + 0.5) / cols, (r + 0.5) / rows) for r in range(rows) for c in range(cols)], np.float32)


def _grid_hex(colour_grid: np.ndarray) -> list:
    rgb = cv2.cvtColor(colour_grid, cv2.COLOR_LAB2RGB)
    return [[f"#{int(p[0]):02x}{int(p[1]):02x}{int(p[2]):02x}" for p in row] for row in rgb]


def _strength(points: int) -> str:
    if points >= STRONG_POINTS:
        return "strong"
    if points >= CONFIRMED_POINTS:
        return "confirmed"
    if points >= WEAK_POINTS:
        return "weak"
    return "none"


def explain_match(upload: Upload, similarity: float, stored: Optional[dict], sscd_model,
                  reference_vector: np.ndarray, feature_map) -> dict:
    """Why one registry match is a match. `stored` is the entry's saved
    feature arrays (None for entries registered before this existed)."""
    if stored is None:
        return {
            "available": False,
            "reason": "This entry was registered before the registry kept distinctive points, so there is "
                      "nothing to compare point by point. Registering it again would enable this.",
        }

    reference = lf.LocalFeatures.from_arrays(stored)
    meta = stored_meta(stored)
    ref_things = group_things(meta["areas"])
    verification = lf.verify(upload.features, reference, upload.mirrored())
    shared = verification.shared_points
    explanation = {
        "available": True,
        "similarity": round(float(similarity), 4),
        "shared_points": shared,
        "candidate_pairs": verification.candidates,
        "strength": _strength(shared),
        "namer": meta.get("namer"),
    }
    centres = _cell_centres(feature_map)

    if shared < CONFIRMED_POINTS:
        # Without a trustworthy placement the original's areas can't be
        # located inside the upload -- so the score is split over the
        # upload's OWN areas instead, and the page says plainly that the
        # points didn't confirm the match.
        area_to_thing = {a: t for t, thing in enumerate(upload.things) for a in thing["areas"]}
        owners = [area_to_thing[lf.area_index(min(x, 0.9999), min(y, 0.9999))] for x, y in centres]
        shares, total = content_registry.similarity_shares(sscd_model, feature_map, reference_vector, owners)
        parts = [{"name": thing["name"], "named": thing["named"], "positions": thing["positions"],
                  "share": round(shares.get(t, 0.0), 4)} for t, thing in enumerate(upload.things)]
        explanation.update(confirmed=False, upload_parts=sorted(parts, key=lambda p: -p["share"]),
                           shares_total=round(total, 4))
        return explanation

    geometry = lf.describe_geometry(verification, upload.features, reference)
    colour = lf.compare_colour(upload.frame, verification, reference)

    wq, hq = upload.features.frame_size
    wr, hr = reference.frame_size
    transform = verification.transform
    frame_points = centres * np.float32([wq, hq])
    if verification.mirrored:
        frame_points[:, 0] = wq - frame_points[:, 0]
    in_original = frame_points @ transform[:, :2].T + transform[:, 2]
    area_to_thing = {a: t for t, thing in enumerate(ref_things) for a in thing["areas"]}
    owners = []
    for x, y in in_original:
        area = lf.area_index(x / wr, y / hr)
        owners.append("outside" if area is None else area_to_thing[area])
    shares, total = content_registry.similarity_shares(sscd_model, feature_map, reference_vector, owners)

    ref_points = reference.points[verification.ref_idx]
    point_things = [area_to_thing[lf.area_index(min(x / wr, 0.9999), min(y / hr, 0.9999))] for x, y in ref_points]
    per_thing = Counter(point_things)
    things = [{
        "name": thing["name"],
        "named": thing["named"],
        "positions": thing["positions"],
        "confidence": thing["confidence"],
        "shared_points": int(per_thing.get(t, 0)),
        "share": round(shares.get(t, 0.0), 4),
        "in_upload": t in shares,
        "index": t,
    } for t, thing in enumerate(ref_things)]

    query_points = (upload.mirrored() if verification.mirrored else upload.features).points[verification.query_idx].copy()
    if verification.mirrored:
        query_points[:, 0] = wq - query_points[:, 0]
    picks = np.unique(np.linspace(0, shared - 1, min(shared, FIGURE_LINES)).astype(int))
    lines = [[round(float(query_points[i, 0]) / wq, 4), round(float(query_points[i, 1]) / hq, 4),
              round(float(ref_points[i, 0]) / wr, 4), round(float(ref_points[i, 1]) / hr, 4), int(point_things[i])]
             for i in picks]

    explanation.update(
        confirmed=True,
        geometry=geometry,
        colour=colour,
        common=sorted([t for t in things if t["in_upload"]], key=lambda t: -t["share"]),
        only_in_original=[t for t in things if not t["in_upload"]],
        not_from_original_share=round(shares.get("outside", 0.0), 4),
        shares_total=round(total, 4),
        figure={
            "original_aspect": round(wr / hr, 4),
            "original_points": [[round(float(x) / wr, 4), round(float(y) / hr, 4)]
                                for x, y in reference.points[:FIGURE_REFERENCE_POINTS]],
            "colour_grid": _grid_hex(reference.colour_grid),
            "area_names": [a["name"] or a["position"] for a in meta["areas"]],
            "area_things": [area_to_thing[a] for a in range(9)],
            "lines": lines,
            "footprint": geometry["footprint"],
        },
    )
    return explanation
