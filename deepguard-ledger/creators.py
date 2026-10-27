"""
Creator identities — one Ed25519 key pair per person who registers media.

In plain words: Deep-Guard's own signature on a block says "Deep-Guard
wrote this page". It does not say WHO asked for a record to be written.
So every creator gets their own key pair, and every record they file
(registration, revocation, dispute) carries the creator's signature too.
Anyone can then check that a revocation really came from the person who
registered the original, and not from someone else.

Honest simplification (the blueprint's section 5.2 asks for it to be
said out loud): this is a SIMULATED public-key infrastructure. The
private keys are kept by this server in ledger_data/creators/, the way a
custodial wallet would hold them, and nobody checks a creator's real-world
identity before handing out a key. A real deployment needs creators to
hold their own keys (on their phone or camera) and an accredited
certificate authority to vouch for who they are — an open problem even
for C2PA.
"""

import json
import re
import threading
from pathlib import Path

import ledger_core

HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{1,39}$")


def slug_of(handle: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", handle.strip().lower()).strip("-")


def clean_handle(handle: str) -> str:
    handle = " ".join((handle or "").split())
    if not HANDLE_PATTERN.match(handle):
        raise ValueError("A creator name needs 2-40 characters: letters, digits, spaces, dots, dashes or underscores.")
    return handle


class CreatorBook:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "creators"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _paths(self, handle: str) -> tuple[Path, Path]:
        slug = slug_of(handle)
        return self.dir / f"{slug}.json", self.dir / f"{slug}.private.pem"

    def get(self, handle: str):
        public_path, _ = self._paths(handle)
        if not public_path.exists():
            return None
        return json.loads(public_path.read_text(encoding="utf-8"))

    def get_or_create(self, handle: str) -> tuple[dict, bool]:
        """Returns (public identity, created_now)."""
        handle = clean_handle(handle)
        with self.lock:
            existing = self.get(handle)
            if existing is not None:
                return existing, False
            private_pem, public_pem = ledger_core.generate_keypair()
            public_path, private_path = self._paths(handle)
            identity = {
                "handle": handle,
                "creator_id": slug_of(handle),
                "public_key": public_pem.decode("ascii"),
                "key_id": ledger_core.key_id(public_pem),
                "created_at": ledger_core.now_iso(),
            }
            private_path.write_bytes(private_pem)
            public_path.write_text(json.dumps(identity, indent=1), encoding="utf-8")
            return identity, True

    def sign_record(self, handle: str, record: dict) -> dict:
        """Adds the creator's key and signature to a record."""
        identity = self.get(clean_handle(handle))
        if identity is None:
            raise KeyError(f"No creator called {handle!r}")
        _, private_path = self._paths(handle)
        signed = {**record, "creator_id": identity["creator_id"], "creator_handle": identity["handle"],
                  "creator_public_key": identity["public_key"], "creator_key_id": identity["key_id"]}
        signed["creator_signature"] = ledger_core.sign_hex(
            private_path.read_bytes(), ledger_core.canonical_bytes(ledger_core.unsigned_record(signed)))
        return signed

    def list(self) -> list[dict]:
        return sorted((json.loads(p.read_text(encoding="utf-8")) for p in self.dir.glob("*.json")),
                      key=lambda c: c["created_at"])
