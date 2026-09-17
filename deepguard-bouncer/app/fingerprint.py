"""
Deep-Guard Stage 2 -- Fingerprint Engine.

Every uploaded file gets TWO fingerprints, because they answer different
questions and neither one alone is enough:

  1. EXACT fingerprint (SHA-256 Merkle root). Hashes the raw bytes. Changing
     a single bit produces a completely different hash. Answers: "is this
     byte-for-byte the exact file that was registered?"

  2. VISUAL fingerprint (perceptual hash / pHash). Hashes the coarse visual
     structure -- shapes, edges, brightness pattern -- not the bytes. Answers:
     "does this still look like the same image/video, even after it's been
     re-compressed by WhatsApp/Twitter/Instagram?"

A record built from both, plus a timestamp, is then signed with Ed25519 so
that WHO registered it (and that the record itself hasn't been altered
since) can be verified independently by anyone with the public key.

Note on scope: this module only produces and verifies fingerprints/signatures
for a single upload. It does not store anything or search a registry of past
records -- that's the ledger (Stage 3), built separately on top of this.
"""

import hashlib
import json
from datetime import datetime, timezone

import imagehash
from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

CHUNK_SIZE = 1024 * 1024  # 1 MB per Merkle leaf

# 16 -> a 256-bit perceptual hash, per the project blueprint's own
# recommendation (the default 8x8/64-bit hash leaves too small a gap
# between "same image" and "different image" scores).
PHASH_SIZE = 16


# ---------------------------------------------------------------------------
# Exact-integrity fingerprint: SHA-256 Merkle root over file chunks
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def compute_merkle_leaves(file_bytes: bytes, chunk_size: int = CHUNK_SIZE) -> list[bytes]:
    """Splits the file into fixed-size chunks and hashes each one. Chunking
    (rather than one hashlib.sha256(whole_file) call) is what lets large
    video files be hashed without holding a second full copy in memory, and
    is the standard first step toward Merkle inclusion proofs later."""
    if not file_bytes:
        return [_sha256(b"")]
    return [_sha256(file_bytes[i : i + chunk_size]) for i in range(0, len(file_bytes), chunk_size)]


def merkle_root(leaves: list[bytes]) -> bytes:
    """Standard binary Merkle tree: repeatedly hash adjacent pairs until one
    root hash remains. An unpaired node at the end of a level is promoted
    to the next level unchanged rather than duplicated -- simpler, and
    sufficient here since this root is used for tamper-evidence, not for
    cross-record consensus (duplicate-last-node has known edge-case forgery
    issues in that stricter setting, which doesn't apply to a single file)."""
    level = leaves
    if not level:
        return _sha256(b"")
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                next_level.append(_sha256(level[i] + level[i + 1]))
            else:
                next_level.append(level[i])
        level = next_level
    return level[0]


def compute_file_fingerprint(file_bytes: bytes) -> dict:
    """The exact-integrity half of a record. Changing even one byte of the
    input changes this root completely."""
    leaves = compute_merkle_leaves(file_bytes)
    return {
        "sha256_merkle_root": merkle_root(leaves).hex(),
        "chunk_count": len(leaves),
        "file_size_bytes": len(file_bytes),
    }


# ---------------------------------------------------------------------------
# Visual fingerprint: perceptual hash (survives compression/re-encoding)
# ---------------------------------------------------------------------------

def compute_image_phash(image: Image.Image) -> str:
    """A 256-bit hash of the image's coarse visual structure (DCT-based
    pHash). Two images that look the same to a human -- even after JPEG
    re-compression, resizing, or a brightness tweak -- produce nearly
    identical hashes. Deliberately NOT sensitive to exact bytes; that's
    compute_file_fingerprint's job, not this one."""
    return str(imagehash.phash(image.convert("RGB"), hash_size=PHASH_SIZE))


def compute_video_phash(frames: list[Image.Image]) -> list[str]:
    """One perceptual hash per sampled frame, kept as a list rather than
    collapsed into a single hash so temporal structure survives -- two
    clips that happen to share one frame but differ everywhere else
    shouldn't be indistinguishable from an exact match."""
    return [compute_image_phash(frame) for frame in frames]


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Counts differing bits between two hex-encoded perceptual hashes.
    0 = identical. Similarity thresholds (see blueprint) are tuned against
    this number, never against exact string equality -- pHash is designed
    to move slightly even between two "matching" images.

    imagehash's `-` operator returns a numpy integer, not a native Python
    int, despite behaving like one everywhere it was previously used
    (printing, arithmetic). The `int()` cast matters the moment this value
    needs to be JSON-serialized (e.g. returned from a FastAPI endpoint),
    since numpy integers aren't natively serializable.
    """
    return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))


def video_similarity(frames_a: list[str], frames_b: list[str]) -> float:
    """Mean Hamming distance across the shorter of the two frame-hash lists,
    aligned from the start. A first approximation: proper alignment for
    clips of different lengths/frame rates/start offsets is future work for
    the ledger's search step, not needed to prove the core mechanism here."""
    if not frames_a or not frames_b:
        return float("inf")
    n = min(len(frames_a), len(frames_b))
    distances = [hamming_distance(frames_a[i], frames_b[i]) for i in range(n)]
    return sum(distances) / n


# ---------------------------------------------------------------------------
# Signing: proves WHO registered a fingerprint, and that the record hasn't
# been altered since. Ed25519 -- fast, small keys/signatures, and (unlike
# RSA) has no parameters to misconfigure.
# ---------------------------------------------------------------------------

def generate_keypair() -> tuple[bytes, bytes]:
    """Returns (private_key_pem, public_key_pem). In a real deployment each
    registrant would hold their own private key; for this demo, Deep-Guard
    holds one signing identity representing "the Deep-Guard registration
    service attests this fingerprint was computed by us, right now." That
    simplification -- and what a real accredited signer identity would
    additionally require -- belongs in the report, not hidden in code."""
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _canonical_record_bytes(record: dict) -> bytes:
    """Deterministic byte encoding so signing and verifying always hash the
    same bytes regardless of the dict's key insertion order."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_record(private_key_pem: bytes, record: dict) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    signature = private_key.sign(_canonical_record_bytes(record))
    return signature.hex()


def verify_signature(public_key_pem: bytes, record: dict, signature_hex: str) -> bool:
    """`record` may be passed either with or without its own "signature" key
    (e.g. the full record returned by build_and_sign_record) -- the
    signature was always computed over the record's other fields only, so
    that key is stripped here rather than requiring the caller to remember to."""
    public_key = serialization.load_pem_public_key(public_key_pem)
    record_without_signature = {k: v for k, v in record.items() if k != "signature"}
    try:
        public_key.verify(bytes.fromhex(signature_hex), _canonical_record_bytes(record_without_signature))
        return True
    except Exception:  # noqa: BLE001 -- any failure means "does not verify"
        return False


def build_and_sign_record(private_key_pem: bytes, public_key_pem: bytes, **fields) -> dict:
    """Assembles a record from the given fields plus a UTC timestamp and the
    signer's public key, signs it, and returns the complete record with its
    signature attached. This is the one function main.py actually calls."""
    record = {
        **fields,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "signer_public_key": public_key_pem.decode("ascii"),
    }
    record["signature"] = sign_record(private_key_pem, record)
    return record
