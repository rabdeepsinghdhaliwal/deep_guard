# Trust lists for reading Content Credentials

A C2PA signature only says *who* signed when the signer's certificate chains
to a root you trust. These are the public lists Deep-Guard trusts, kept here
as pinned snapshots so the check works offline (in a classroom with no
internet) and gives the same answer on every machine.

| File | What | Source | Fetched | SHA-256 |
|---|---|---|---|---|
| `C2PA-TRUST-LIST.pem` | The official C2PA Trust List (claim signers of conformant products: Google, SSL.com, …), 33 certificates | https://github.com/c2pa-org/conformance-public/tree/main/trust-list (commit 64b3f9c8ebb9, 2026-11-04) | 2026-11-05 | `148c1c167e98dc909a0d5793d2feee1022aa60fef5c83a0d6b08f67f4446bc37` |
| `interim-trust-list-anchors.pem` | The Content Authenticity Initiative's interim list used by contentcredentials.org/verify, frozen on 2026-01-01; still valid for content signed under it, 27 certificates | https://contentcredentials.org/trust/anchors.pem | 2026-11-05 | `db57e50b9bd4e4c786cbb91f38fd85e2af6a9d5412438a8cddcc42688df07ae5` |

Deep-Guard's own demo certificate authority is NOT in these files: each
installation makes its own (in `deepguard-ledger/credentials/`, never
committed), and the pages report credentials signed by it as "trusted by this
Deep-Guard server", never as being on a public list.

Refresh with `python tools/update_trust_lists.py` (needs internet).
