"""
Deep-Guard -- one answer format for every layer of the trust check.

The roadmap turns three side-by-side engines into one layered check:

    0  Content Credentials   the maker's own signed record, if the file has one
    1  Watermarks            invisible marks some generators embed
    2  Detectors             statistical guesses (EfficientNet + a second one)
    3  Known-content match   the registry, plus reverse image search
    4  Trust report          all of the above on one sealed page

For the report to put the layers side by side, order them strongest first,
and notice when two of them disagree, every layer has to answer in the
same shape -- the way witnesses in a court all give evidence under oath
in the same form, whatever their expertise. That shape is `Evidence`.

Step 0 defines it (and the exam harness already records the detector's
answers in it); Step 1 wraps today's engines in it; every later layer is
built to return it.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    FOUND = "found"            # the layer has something to report
    NOT_FOUND = "not_found"    # it ran and found nothing (e.g. no credentials) -- proves nothing either way
    NOT_RUN = "not_run"        # switched off or not applicable (e.g. reverse search left off)
    ERROR = "error"            # it tried and failed


class PointsTo(str, Enum):
    AI = "ai"
    REAL = "real"
    NEITHER = "neither"        # e.g. "this is a copy of registered image X" says nothing about AI


class Strength(str, Enum):
    """How much a finding should weigh, strongest first. The report orders
    evidence by this, never by which layer happened to run first."""
    PROOF = "proof"            # cryptographic: a valid signature from a signer on a trust list
    STRONG = "strong"          # hard to produce by accident: a decoded watermark, a dated earlier copy
    INDICATION = "indication"  # statistical: a detector's (calibrated) probability
    NONE = "none"              # nothing to weigh (not found / not run / error)


STRENGTH_ORDER = [Strength.PROOF, Strength.STRONG, Strength.INDICATION, Strength.NONE]
LAYER_NAMES = {0: "Content Credentials", 1: "Watermarks", 2: "Detectors", 3: "Known-content match"}


@dataclass
class Evidence:
    layer: int                   # 0-3, see LAYER_NAMES
    check: str                   # machine name: "c2pa", "trustmark", "efficientnet", "registry", ...
    title: str                   # what a person sees: "Our AI-image detector"
    status: Status
    points_to: PointsTo
    strength: Strength
    claim: str                   # one plain sentence: "Signed by OpenAI: made with gpt-image-1."
    confidence: Optional[float] = None   # calibrated P(claim) for statistical checks, else None
    details: dict = field(default_factory=dict)   # the layer's own facts, for the full view
    limits: list = field(default_factory=list)    # what this result can NOT show, in plain words

    def validate(self) -> "Evidence":
        """Rejects combinations that would make the report say something
        untrue -- e.g. "strong evidence" from a check that never ran."""
        problems = []
        if self.layer not in LAYER_NAMES:
            problems.append(f"layer must be one of {sorted(LAYER_NAMES)}, got {self.layer}")
        if not isinstance(self.status, Status):
            problems.append("status must be a Status")
        if not isinstance(self.strength, Strength):
            problems.append("strength must be a Strength")
        if not isinstance(self.points_to, PointsTo):
            problems.append("points_to must be a PointsTo")
        if self.status != Status.FOUND and self.strength != Strength.NONE:
            problems.append(f"a '{self.status.value}' result cannot carry '{self.strength.value}' strength")
        if self.status == Status.FOUND and self.strength == Strength.NONE:
            problems.append("a 'found' result needs a strength above 'none'")
        if self.status != Status.FOUND and self.points_to != PointsTo.NEITHER:
            problems.append("only a 'found' result can point towards AI or real")
        if self.strength == Strength.PROOF and self.layer != 0:
            problems.append("only a verified signature (layer 0) counts as proof")
        if self.strength == Strength.INDICATION and self.confidence is None:
            problems.append("a statistical indication needs its confidence")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence must be between 0 and 1")
        if not self.claim.strip():
            problems.append("claim must be a sentence a person can read")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("status", "points_to", "strength"):
            d[k] = d[k].value
        d["layer_name"] = LAYER_NAMES[self.layer]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(layer=d["layer"], check=d["check"], title=d["title"], status=Status(d["status"]),
                   points_to=PointsTo(d["points_to"]), strength=Strength(d["strength"]), claim=d["claim"],
                   confidence=d.get("confidence"), details=d.get("details") or {},
                   limits=d.get("limits") or []).validate()


def strongest_first(items: list) -> list:
    """Orders evidence the way the report will read it: by strength, then by
    layer (a maker's signature before a watermark before a detector)."""
    return sorted(items, key=lambda e: (STRENGTH_ORDER.index(e.strength), e.layer))


def disagreements(items: list) -> list:
    """Pairs of found evidence pointing opposite ways -- the report must show
    these, not average them away (e.g. credentials say "camera" while a
    watermark says "AI": see Authenticated Contradictions, 2026)."""
    found = [e for e in items if e.status == Status.FOUND and e.points_to != PointsTo.NEITHER]
    return [(a, b) for i, a in enumerate(found) for b in found[i + 1:] if a.points_to != b.points_to]
