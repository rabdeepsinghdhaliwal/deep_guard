"""
Deep-Guard Stage 4 — a hidden ID in the pixels (TrustMark).

In plain words: the ledger finds a copy by its fingerprints, but metadata is
stripped the moment a photo goes through WhatsApp, Instagram or a
screenshot, and a heavily edited copy can drift far from its fingerprints.
So at registration Deep-Guard hides a random 61-bit ID inside the picture's
pixels, invisibly, and hands the creator that "credentialed copy" to share.
Whatever later happens to a copy of it (re-saved, resized, screenshotted,
cropped), reading the pixels gives the ID back, and the ID points straight at
its ledger record.

The watermark is TrustMark (Bui, Agarwal & Collomosse, Adobe Research,
arXiv 2311.18297), the open watermark behind Adobe's "durable Content
Credentials": variant Q, the one TrustMark's own C2PA example uses, with
BCH_5 error correction, so the ID survives up to 5 wrong bits. That example
writes the ID into a manifest with the soft-binding algorithm name
`com.adobe.trustmark.Q`; phase 3 does the same.

An ID alone is never trusted. It can be read by chance from an unmarked
picture (in the October feasibility test, 1 in 120 unmarked pictures gave
one), and a watermark can be lifted from one picture onto another. So
ledger_app.py only accepts an ID when the picture's content also matches the
record it names (SSCD similarity >= 0.40), the "fingerprint" pillar of
durable credentials. TrustMark also ships a remover model; anyone with it can
strip the mark, which is why the fingerprints stay the fallback.
"""

import secrets
import threading
from typing import Optional

from PIL import Image

ALG = "com.adobe.trustmark.Q"   # the C2PA soft-binding algorithm name for this variant
SCHEMA = "BCH_5"                # 61 data bits, corrects up to 5 bit errors
SCHEMA_CODE = 1                 # TrustMark's own number for BCH_5, written before the ID in a soft binding
ID_BITS = 61


def new_id() -> str:
    """A random 61-bit ID, as the 0/1 string TrustMark's binary mode takes."""
    return format(secrets.randbits(ID_BITS), f"0{ID_BITS}b")


def id_hex(bits: str) -> str:
    """The same ID as 16 hex digits, the way records and pages show it."""
    return f"{int(bits, 2):016x}"


def bits_from_hex(value: str) -> str:
    return format(int(value, 16), f"0{ID_BITS}b")


class Watermarker:
    """TrustMark, loaded on first use (about 65 MB of weights; the first
    run on a new machine downloads them from Adobe's server). One lock,
    because the model is shared by every request."""

    def __init__(self):
        self._tm = None
        self._lock = threading.Lock()
        self.load_error: Optional[str] = None

    def _model(self):
        if self._tm is None:
            from trustmark import TrustMark  # heavy import: only when a watermark is needed
            self._tm = TrustMark(verbose=False, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5,
                                 loadRemover=False, device="cpu")
            if self._tm.schemaCapacity() != ID_BITS:
                raise RuntimeError(f"TrustMark BCH_5 holds {self._tm.schemaCapacity()} bits, expected {ID_BITS}")
        return self._tm

    def warm_up(self) -> None:
        """Load the model now (e.g. in a background thread at startup), so
        the first registration doesn't pay for it."""
        try:
            with self._lock:
                self._model()
        except Exception as exc:  # noqa: BLE001 — additive: the ledger works without watermarks
            self.load_error = f"{exc.__class__.__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.load_error is None

    def embed(self, image: Image.Image, bits: str) -> Image.Image:
        with self._lock:
            return self._model().encode(image.convert("RGB"), bits, MODE="binary")

    def read(self, image: Image.Image) -> Optional[str]:
        """The hidden ID as 0/1 bits, or None if no intact Deep-Guard-sized
        TrustMark ID is in the picture."""
        with self._lock:
            bits, present, schema = self._model().decode(image.convert("RGB"), MODE="binary")
        if not present or len(bits) != ID_BITS:
            return None
        return bits
