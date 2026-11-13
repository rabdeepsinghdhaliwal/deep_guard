"""
Refresh the pinned C2PA trust lists in deepguard-ledger/trust/ (needs internet).

    python tools/update_trust_lists.py

Downloads the official C2PA Trust List and the (frozen) interim list, checks
each is a bundle of PEM certificates, writes them over the pinned copies and
prints their SHA-256 and certificate counts for trust/README.md. Restart the
server afterwards so it loads the new lists.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

TRUST = Path(__file__).resolve().parents[1] / "trust"
SOURCES = {
    "C2PA-TRUST-LIST.pem":
        "https://raw.githubusercontent.com/c2pa-org/conformance-public/main/trust-list/C2PA-TRUST-LIST.pem",
    "interim-trust-list-anchors.pem": "https://contentcredentials.org/trust/anchors.pem",
}


def main() -> int:
    for name, url in SOURCES.items():
        request = urllib.request.Request(url, headers={"User-Agent": "deepguard-ledger/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(2_000_000)
        except OSError as exc:
            print(f"{name}: could not download ({exc}); the pinned copy is kept.")
            return 2
        count = data.count(b"-----BEGIN CERTIFICATE-----")
        if count == 0:
            print(f"{name}: the download holds no certificates; the pinned copy is kept.")
            return 1
        (TRUST / name).write_bytes(data)
        print(f"{name}: {count} certificates, SHA-256 {hashlib.sha256(data).hexdigest()}")
    print("Update the table in trust/README.md with these values, then restart the server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
