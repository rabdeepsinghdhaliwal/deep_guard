# Deep-Guard Ledger (Stage 3)

The "blockchain part" of Deep-Guard: a **signed, hash-chained, append-only
provenance ledger** with independent witnesses, built beside the bouncer
without changing a single bouncer file.

> **In one line:** the bouncer *guesses* whether a picture is AI-made; the
> ledger *proves* where a picture came from — if its creator wrote it down
> here first — and only falls back to the guess when there is no record.

No Solidity, no Ethereum, no tokens, no mining. It follows the project
blueprint's own decision (section 3): the property we need is a
tamper-evident, append-only, independently checkable record — the
data-structure heart of a blockchain (hash-linked blocks, Merkle trees,
signatures) — plus witnesses so that no single party, not even Deep-Guard,
can silently rewrite it.

---

## Run it

From the repository root, once the bouncer's `venv` is set up (see the
main README's Quick start — the ledger needs no extra packages):

```bash
cd deepguard-ledger
../deepguard-bouncer/venv/Scripts/python.exe run_deepguard.py   # Windows
../deepguard-bouncer/venv/bin/python run_deepguard.py           # macOS/Linux
```

This starts **three** things: the main server on port 8000 (the bouncer
plus the ledger), and two witnesses on ports 8101 and 8102. Wait for
`Application startup complete`, then open:

| Page | What it is |
|---|---|
| http://127.0.0.1:8000/ | The original analysis page — now with a **provenance card** under every result and **Ledger / Explorer** in the menu |
| http://127.0.0.1:8000/ledger | Register a file, verify a file (four verdicts), revoke / dispute / resolve |
| http://127.0.0.1:8000/explorer | The block explorer: the whole chain drawn, every check redone in your browser, witnesses, tamper lab, receipt checker |
| /registry, /bias-map | Unchanged bouncer pages (plus the two menu links) |

`Ctrl+C` stops all three. The old way (`uvicorn main:app` inside
`deepguard-bouncer/app`) still starts the original bouncer exactly as before,
without the ledger.

---

## What one block looks like

```
block #3
  index          3
  timestamp      2026-10-27T15:02:11.481+00:00
  previous_hash  9f2c…   <- block #2's block_hash: this is the chain
  merkle_root    41ab…   <- one hash summarising every record inside
  record_count   2
  signer_key_id  7e0d…   <- which key signed it (Deep-Guard's)
  block_hash     c88e…   = SHA-256 of the six fields above
  signature      …       = Deep-Guard's Ed25519 signature of block_hash
  records        [ registration, dispute, … ]   each signed by its creator
```

Change one letter anywhere in an old block and its Merkle root or block
hash stops matching; recompute them and the signature fails; re-sign it
too and the *next* block's `previous_hash` no longer matches. Re-sign the
whole rest of the chain with the real key and the chain looks perfect — but
the creators' signatures (for edits) and the witnesses (for deletions,
reordering and backdating) still catch it.

## What a registration record holds

| Field | Meaning |
|---|---|
| `sha256_merkle_root` | exact fingerprint — changes if one byte changes (Stage 2, reused) |
| `phash` / `phash_frames` | 256-bit "looks-like" fingerprint (per frame for videos) — survives re-compression |
| `content_vector_sha256` | hash of the SSCD content vector (512 numbers kept beside the ledger in `ledger_data/content_vectors/`) — finds crops |
| `declared_origin` | `camera`, `edited` or `ai-generated` (the creator's own claim) |
| `detector_reading` | what the bouncer's AI detector said at registration — written beside the claim, not used to judge it |
| `creator_handle`, `creator_public_key`, `creator_signature` | who filed it, signed with their own Ed25519 key |

The picture itself is never stored.

## The four verdicts (provenance first, prediction last)

| # | When | Verdict |
|---|---|---|
| 1 | exact SHA-256 match | **Authentic & unaltered** |
| 2 | no exact match, but pHash (≤ 32 of 256 bits) or SSCD content (≥ 0.40) match | **Visually authentic, file modified** — re-saved, compressed, cropped |
| 3 | nothing matches | **No provenance found** → the AI detector runs, labelled "a prediction, not proof" |
| 4 | match to a record declared `ai-generated` | **Registered as AI-generated** — honest synthetic content |

Revoked registrations no longer count; disputed ones are flagged; a match
inside a block that fails verification comes with a warning.

---

## Everything that was added

| Piece | File | What it does |
|---|---|---|
| Ledger core | `ledger_core.py` | canonical JSON, SHA-256, Merkle trees + inclusion proofs (RFC 6962 style), blocks, hash chain, Ed25519 block signatures, append-only files, full chain verification with plain-words reasons, receipts, tamper simulations |
| Creator keys | `creators.py` | one Ed25519 key pair per creator; every record they file is signed by them (a simulated PKI — see limits) |
| Witnesses | `witness.py` | independent processes with their own keys and notebooks; check each new block, co-sign it, and raise an alarm if Deep-Guard ever shows a different history |
| Server | `ledger_app.py` | adds the pages and `/api/ledger/*` to the bouncer's own app; injects the menu links + provenance card into the old pages at serve time (files untouched) |
| Ledger page | `static/ledger.html`, `ledger.js` | register, verify, revoke, dispute, resolve |
| Explorer | `static/explorer.html`, `explorer.js` | visual chain, block + Merkle tree views, record chain-of-trust, search, witnesses, tamper lab, receipt checker, keys |
| In-browser checks | `static/dgcrypto.js` | the same hashing, Merkle and signature checks re-implemented with the browser's Web Crypto |
| Bridge | `static/bridge.js` | "Ledger / Explorer" menu links; the provenance card under analysis results |
| Offline checker | `tools/verify_ledger.py` | checks the whole chain or one receipt with no server, using a public key you trust |
| Launcher | `run_deepguard.py` | starts the server and both witnesses with one command |
| Tests | `tests/` | 47 tests: core maths and every attack, plus the full app with real models, real witness servers, images and video |

### API (all under `/api/ledger/`)

`summary`, `chain`, `verify-chain`, `register`, `seal`, `verify`, `revoke`,
`dispute`, `resolve`, `record/{id}`, `receipt/{id}`, `merkle/{index}`,
`creators`, `witnesses`, `witnesses/sync`, `attacks`, `tamper-lab`.

---

## Check it yourself

* **In the browser:** Explorer → *Re-check in my browser*. Every hash,
  Merkle root, link and signature is recomputed by the page itself.
* **The live tamper demo:** open `ledger_data/chain.jsonl` in VS Code,
  change one letter in an old block's title, save, press *Re-check in my
  browser* → that block turns red with the exact reason. Undo → green.
* **Tamper lab:** four attacks run on a copy of the chain (real files
  untouched) — edit a record; edit and recompute hashes without the key;
  delete a block; an insider rewrite with the real key (only the witnesses
  catch that one).
* **Offline, no server:**

  ```bash
  ../deepguard-bouncer/venv/Scripts/python.exe tools/verify_ledger.py chain
  ../deepguard-bouncer/venv/Scripts/python.exe tools/verify_ledger.py receipt deepguard-receipt-xxxx.json ../deepguard-bouncer/models/demo_signer_public.pem
  ```

* **Tests:** `../deepguard-bouncer/venv/Scripts/python.exe -m pytest tests -q`
  (needs `pytest` and `httpx` in the venv: `pip install pytest httpx`).

To start a fresh ledger, stop the server and delete the `ledger_data/`
folder (it also holds the witnesses' notebooks and the creators' keys).

---

## Honest limits

* **Simulated identities.** Creators' private keys are kept by this server
  (like a custodial wallet) and nobody checks who a creator really is. A
  real system needs creators to hold their own keys and an accredited
  certificate authority — an open problem even for C2PA.
* **Witnesses run on the same machine.** They are separate processes with
  separate keys and files, which is enough to demonstrate the mechanism;
  real witnesses would be run by other organisations on other machines.
* **First registration wins.** The ledger proves *who wrote it down first*,
  not who really created it. That is what disputes are for.
* **The detector fallback is the bouncer's detector**, with all of its
  measured weaknesses (about 58% of today's AI images caught, about 14%
  false alarms on real photos). That is exactly why it comes last.
* **Video matching** compares frame hashes in order; trimmed or re-timed
  clips may not match (the bouncer's `video_similarity` notes this too).
