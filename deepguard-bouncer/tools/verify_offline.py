"""
Deep-Guard -- check a signed record offline, with nothing but a public key.

Every fingerprint the server produces (and every Content Registry entry) is
a record sealed with the server's Ed25519 private key (fingerprint.py). The
"Verify independently" button on the result page asks the SAME server that
made the seal to check it: a fair demonstration of the mechanism, but
self-verification. The point of publishing the public key is that nobody
has to trust the server at all -- anyone holding the record and the key can
check the seal themselves. This script is that check: no Deep-Guard server,
no internet, no Deep-Guard code, only the `cryptography` package.

Run from deepguard-bouncer/:
    python tools/verify_offline.py saved.json models/demo_signer_public.pem

`saved.json` can be a bare record, or a server answer saved as-is: from
/api/analyze (its "fingerprint"), /api/registry/register, or
/api/registry/check (every match is checked). One way to save one:
    curl -F "file=@photo.jpg" http://127.0.0.1:8000/api/analyze -o saved.json
(in Windows PowerShell, type curl.exe -- plain `curl` is a different command
there).

THE RULE THAT MATTERS: the public key must come from a source you trust --
the repository, the key shown in the page footer, the signer in person --
never from the record itself. A record carries its signer's public key for
convenience, but a forger can sign with their own key and embed THAT; a
record checked against the key it carries proves nothing (pinned by
test_a_forger_who_embeds_their_own_key_fails).

What a valid seal proves: the record has not changed since it was signed,
and it was signed by whoever holds the matching private key. What it does
not prove: that the picture is true, that it isn't AI-generated, or who
created it.

Deliberately standalone, so it duplicates one rule from fingerprint.py: the
exact bytes that get signed (canonical JSON -- keys sorted, no spaces). The
tests in app/tests/test_verify_offline.py check this script against the
server's own signing code and real saved server answers, so the two cannot
drift apart silently.

Exit code: 0 every record valid, 1 a record is invalid, 2 couldn't check.
"""
import argparse
import json
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Keys the server's answers add NEXT TO a signed record: /api/registry/register
# adds distinct_features, /api/registry/check adds similarity, entry_id and
# explanation to each match. They are computed when the server answers, were
# never part of what it signed, and are set aside before checking -- the same
# rule main.py's registry endpoints document. The printout names them, so
# nobody mistakes them for signed facts.
UNSIGNED_EXTRAS = ("distinct_features", "similarity", "entry_id", "explanation")


def canonical_bytes(fields: dict) -> bytes:
    """The exact bytes the server signs (fingerprint._canonical_record_bytes):
    the same data always serialises the same way, whatever the key order."""
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_trusted_key(pem: bytes) -> Ed25519PublicKey:
    """Parses the trusted public key. Tolerates what copy-pasting or
    Windows Notepad adds (a byte-order mark, blank lines); refuses anything
    that isn't an Ed25519 key, since no other kind could have signed."""
    pem = pem.lstrip(b"\xef\xbb\xbf").strip() + b"\n"
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("that key is not an Ed25519 public key; Deep-Guard signs with Ed25519")
    return key


def signed_fields(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "signature" and k not in UNSIGNED_EXTRAS}


def verify(record: dict, trusted_public_key_pem: bytes) -> bool:
    """True only if `record`'s signature was made over exactly its signed
    fields by the private key matching `trusted_public_key_pem`."""
    key = load_trusted_key(trusted_public_key_pem)
    signature_hex = record.get("signature")
    if not isinstance(signature_hex, str):
        return False
    try:
        key.verify(bytes.fromhex(signature_hex), canonical_bytes(signed_fields(record)))
        return True
    except (ValueError, InvalidSignature):  # not hex / seal doesn't fit these bytes
        return False


def signed_records(document) -> list:
    """(label, record) for every signed record in a saved file: a bare
    record, or the answer of /api/analyze, /api/registry/register or
    /api/registry/check, saved exactly as the server sent it."""
    if not isinstance(document, dict):
        return []
    if "signature" in document:
        return [("record", document)]
    if isinstance(document.get("fingerprint"), dict):
        return [("fingerprint", document["fingerprint"])]
    matches = document.get("matches")
    if isinstance(matches, list):
        return [(f"match {i}", m) for i, m in enumerate(matches, 1) if isinstance(m, dict)]
    return []


def _raw(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def names_other_key(record: dict, trusted_key: Ed25519PublicKey) -> bool:
    """True if the record's own signer_public_key is a different key from
    the trusted one -- only ever an explanatory note, never a reason to
    trust anything."""
    embedded = record.get("signer_public_key")
    if not isinstance(embedded, str):
        return False
    try:
        return _raw(load_trusted_key(embedded.encode("ascii"))) != _raw(trusted_key)
    except Exception:  # noqa: BLE001 -- an unreadable embedded key is simply "not the trusted one"
        return True


def describe(label: str, record: dict, valid: bool, trusted_key: Ed25519PublicKey) -> str:
    """A short plain-text report of one record (ASCII only, so it prints on
    any console; titles are shown JSON-quoted for the same reason)."""
    lines = []
    if valid:
        lines.append(f"VALID    {label}: signed by the key you trust, and unchanged since it was signed.")
    else:
        lines.append(f"INVALID  {label}: changed after it was signed, or not signed by the key you trust.")
    if "title" in record:
        lines.append(f"  title:      {json.dumps(record['title'])}")
    size = record.get("file_size_bytes")
    if isinstance(size, int):
        lines.append(f"  file:       {size:,} byte{'' if size == 1 else 's'}, "
                     f"SHA-256 Merkle root {str(record.get('sha256_merkle_root', ''))[:16]}...")
    if "timestamp_utc" in record:
        lines.append(f"  signed at:  {record['timestamp_utc']} (UTC)")
    extras = [k for k in UNSIGNED_EXTRAS if k in record]
    if extras:
        lines.append(f"  not signed: {', '.join(extras)} (added when the server answered)")
    if names_other_key(record, trusted_key):
        lines.append("  note:       the record names a different signing key than the one you trust.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a Deep-Guard signed record offline, with only the signer's public key.",
        epilog="The key must come from a source you trust, never from the record itself.",
    )
    parser.add_argument("record", type=Path, help="a saved record, or a saved /api/analyze, "
                                                  "/api/registry/register or /api/registry/check answer (.json)")
    parser.add_argument("public_key", type=Path, help="the signer's public key (.pem), from a source you trust")
    args = parser.parse_args(argv)

    try:
        # utf-8-sig: Windows PowerShell's `>` and Out-File save JSON with a byte-order mark
        document = json.loads(args.record.read_text(encoding="utf-8-sig"))
        key_pem = args.public_key.read_bytes()
        trusted_key = load_trusted_key(key_pem)
    except (OSError, ValueError) as exc:  # JSONDecodeError and bad PEM are ValueErrors
        print(f"Can't check: {exc}", file=sys.stderr)
        return 2

    records = signed_records(document)
    if not records:
        if isinstance(document, dict) and document.get("matches") == []:
            reason = "this registry answer has no matches, so it contains nothing signed"
        else:
            reason = ("expected a record with a 'signature', or a saved /api/analyze, "
                      "/api/registry/register or /api/registry/check answer")
        print(f"Can't check {args.record}: {reason}.", file=sys.stderr)
        return 2

    results = [(label, record, verify(record, key_pem)) for label, record in records]
    for label, record, valid in results:
        print(describe(label, record, valid, trusted_key))
    print("A valid seal proves the record is genuine and unchanged -- not that the picture is true, "
          "or who created it.")
    return 0 if all(valid for _, _, valid in results) else 1


if __name__ == "__main__":
    sys.exit(main())
