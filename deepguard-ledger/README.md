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
main README's Quick start), add the ledger's own few packages once:

```bash
pip install -r deepguard-ledger/requirements.txt
```

Then:

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
without the ledger. Add `--public` to let anyone try it from their phone
(see [A live public link](#a-live-public-link-stage-4)).

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
| Tests | `tests/` | 112 tests: core maths and every attack, Bitcoin anchors, Content Credentials, the public link's rules, plus the full app with real models, real witness servers, images and video |

### API (all under `/api/ledger/`)

`summary`, `chain`, `verify-chain`, `register`, `seal`, `verify`, `revoke`,
`dispute`, `resolve`, `record/{id}`, `receipt/{id}`, `merkle/{index}`,
`creators`, `witnesses`, `witnesses/sync`, `attacks`, `tamper-lab`.

---

## Anchored in Bitcoin (Stage 4)

The witnesses stop Deep-Guard from rewriting history on its own; together,
Deep-Guard and its witnesses could still agree to backdate it. An anchor
closes that gap (`anchoring.py`):

- **Explorer → Anchor the newest block to Bitcoin** sends the newest block's
  hash, salted with a random nonce, to four public OpenTimestamps calendars.
  Nothing else leaves the computer. One anchor covers the whole chain up to
  that block, because each block holds the previous block's hash.
- A few hours later, **Check for Bitcoin confirmation** fetches the rest of the
  proof from the calendars and looks the Bitcoin block it names up on two
  independent block explorers (blockstream.info, blockcypher.com; mempool.space
  as a third). The proof's steps must end exactly on that block's Merkle root.
- **Re-check in my browser** does the same with no help from the server: it
  re-hashes the block header, replays the proof's steps with Web Crypto and
  fetches the Merkle root from the explorers itself.
- The anchored "file" is the block's canonical header, whose SHA-256 *is* the
  block hash, so `block-NNNNNN.header.json` plus its `.ots` proof (both
  downloadable) can be checked on opentimestamps.org with no Deep-Guard code.
- **Tamper lab attack 5** rewrites history with the real key and lets the
  witnesses go along with it. The chain and the witnesses both say "fine";
  only an anchor made before the rewrite still disagrees.

Offline: `python tools/verify_ledger.py anchors` (add `--offline` to skip the
Bitcoin lookup). Anchors live in `ledger_data/anchors/`; resetting the ledger
deletes them, so anchor a demo ledger the day before you show it.
Needs `pip install -r deepguard-ledger/requirements.txt`.

**It works on the real network.** Block #2 of a development ledger, anchored
on 2026-11-04 at 16:04 UTC, was in Bitcoin block 974545 seventeen minutes
later (and, through a second calendar, in block 974547). opentimestamps.org
verifies the two downloaded files on its own ("Bitcoin block 974545 attests
existence as of 2026-11-04"), and the proof is kept in `tests/fixtures/anchor/`
so the tests re-check a real Bitcoin proof offline on every run.

## A hidden ID in the pixels (Stage 4)

Metadata is stripped the moment a photo passes through WhatsApp, Instagram or
a screenshot. So registering an image now also makes a **credentialed copy**
(`watermark.py`): the same picture with a random 61-bit ID hidden in its pixels
by TrustMark (Adobe Research's open watermark, variant Q with BCH_5 error
correction). The Ledger page offers it as a download; the creator shares that
copy, not the original. The record keeps the ID and the copy's own
fingerprints, signed by the creator like everything else. The picture itself
is still never stored: the copy is handed over once, in the answer.

Checking a file reads the ID first. A copy of the copy then matches as
**"hidden ID read from the pixels"** even when its pHash has drifted, and a
second person can't register a WhatsApp copy of someone's credentialed photo
as their own. An ID is only believed when the picture's content also matches
the record it names (SSCD ≥ 0.40): a watermark lifted onto a different picture
is read, reported, and not trusted.

Measured on all 55 images in the repository (`tools/measure_watermark.py`,
results in `measurements/watermark_robustness.json`; each image gets a fixed
ID so the run is repeatable):

| The credentialed copy after… | ID read back |
|---|---|
| nothing (as downloaded) | 55 / 55 |
| WhatsApp-style: long side ≤ 1600 px, JPEG quality 75 | 55 / 55 |
| an 80% crop, then WhatsApp-style | 55 / 55 |
| a screenshot: 75% size, PNG | 55 / 55 |
| colour removed | 52 / 55 |
| half size, JPEG quality 70 | 48 / 55 (33 / 35 when the short side is ≥ 512 px) |
| (unmarked originals: an ID found by chance) | 0 / 55 |

The mark measures 39.0 dB PSNR (median, including the JPEG re-save; lowest
34.5 dB on a flat-coloured illustration), which is invisible at normal
viewing. **Limits:** tiny images lose the ID when shrunk further; TrustMark's
own public remover model can strip it, which is why fingerprints stay the
fallback; videos are not watermarked.

## Content Credentials, the CR badge (Stage 4)

C2PA is the industry standard behind the "CR" mark (Adobe, Google, OpenAI,
Nikon, LinkedIn, TikTok…): a signed manifest inside the file saying who made
it, with what, what was done to it and whether AI made it (`credentials.py`).

**Reading.** Every checked file is read for Content Credentials, on the Ledger
page and in the provenance card under every analysis result (and at
`POST /api/credentials/read`). The answer is in plain words: **none**;
**trusted** (the signer is on the official C2PA Trust List or the interim list
contentcredentials.org used, both pinned in `trust/`); **ours** (signed by this
server's own demo CA); **unknown** (signed and unchanged, but no list vouches
for the signer); **expired**; or **altered** (the file or its credentials
changed after signing). On the official C2PA test files it reads a valid test
manifest as "unknown signer", a changed image as "altered", Nikon's 2022 Z 9
sample as "certificate expired", and a file without a manifest as "none".

**Writing.** The credentialed copy is now also signed (Ed25519) by Deep-Guard's
own small certificate authority, with a manifest that says, in C2PA's own
vocabulary: Deep-Guard watermarked it (`c2pa.watermarked`, with the original
as its parent ingredient), the hidden ID it carries (`c2pa.soft-binding`,
`com.adobe.trustmark.Q`, as TrustMark's own C2PA example writes it) and its
ledger record (`com.deepguard.ledger`). A file declared AI-generated is labelled
with the IPTC source type `trainedAlgorithmicMedia`, which other C2PA readers
show as "AI-generated". This is the C2PA's "durable Content Credentials"
design: manifest + watermark + fingerprint. A re-save strips the manifest;
the hidden ID survives and points back to the record.

**Checked with Adobe's own tool.** A credentialed copy of an AI-declared
picture, opened at verify.contentauthenticity.org (2026-11-05), reads as:
"This image was generated with an AI tool", actions "Created" and
"Watermarked — A watermark was added", "Issued by Deep-Guard (student
project)", and, as expected, issuer "Unrecognized".

**Limit.** The demo CA is made once per installation (`credentials/`, never
committed). Our pages say "signed by this Deep-Guard server"; other C2PA
tools, such as contentcredentials.org/verify, will show the manifest but call
the signer unknown, because only certificates from the C2PA conformance
programme are on the official list. `GET /api/credentials/ca` downloads the
demo CA for anyone who chooses to trust it.

## A live public link (Stage 4)

```bash
../deepguard-bouncer/venv/Scripts/python.exe run_deepguard.py --public
```

This starts everything as usual, then opens three Cloudflare **quick
tunnels**, one for the server and one for each witness (`tunnel.py`). A quick
tunnel is a free, temporary `https://…trycloudflare.com` address: no account,
no router settings. `cloudflared` connects *out* to Cloudflare, so it works
even on a college network that lets nobody connect in. The terminal prints the
link and a QR code. Put **http://127.0.0.1:8000/share** on the projector: it
shows the QR code big, with three things to try. Once the link answers from
outside, the terminal says *The public link is live*. `Ctrl+C` closes it, and
the next start gets a new address.

It needs **cloudflared**, a single program from Cloudflare (about 55 MB, not
a pip package). Either run `winget install --id Cloudflare.cloudflared` and
open a new terminal, or save `cloudflared-windows-amd64.exe` from
https://github.com/cloudflare/cloudflared/releases as
`deepguard-ledger/tools/cloudflared.exe` (git ignores it). `--cloudflared PATH`
works too.

**What a visitor can do from their phone:**
* analyse a photo;
* register one under their own name and download its credentialed copy;
* verify the WhatsApp copy of it;
* browse the explorer and run the tamper lab.

The explorer re-checks the chain and the witnesses in the visitor's own
browser. Each witness has a public address of its own, so the phone asks it
directly and doesn't have to take Deep-Guard's word.

**What changes for visitors (`public_mode.py`).** These rules apply only to
requests that came through a tunnel: Cloudflare marks every one with
`CF-Connecting-IP`. The laptop's own browser keeps every feature.

| Rule | Why |
|---|---|
| Uploads up to 15 MB (the laptop keeps 200 MB) | Phone photos fit, and nobody can tie the laptop up with huge videos. |
| Per phone: 12 heavy checks and 6 registrations a minute. Per internet address: 60 and 30. | A whole class on one Wi-Fi shares one address. Phones are told apart by a cookie, and the address gets a bigger share. |
| Heavy checks run one at a time, the presenter's too; a visitor waits up to 40 s | Grad-CAM hooks onto the shared detector model while it works, so two checks side by side could swap heatmaps between people. |
| Off: anchoring, revoking, disputing, resolving, and adding to the older content registry | The server holds every creator's key, so on a public link anyone could type someone else's name. Anchoring calls out to the public calendars. |
| Names used on the laptop are reserved, and a visitor's name stays with the browser that made it | Same reason: otherwise a typed name is all it takes. |
| Witnesses refuse blocks that arrive through their tunnel | Otherwise a visitor could post a forged block and fill the alarm log with false "history was rewritten" alarms. |
| Every page carries a banner, and answers leave out file paths | Visitors know where their picture goes. Checked pictures are never stored; a registration writes fingerprints, never the picture. |

**Before the demo:** register your own demo pictures *before* you show the QR
code. First registration wins, so if a classmate registers one first, yours
would be refused.

**Limits.**
* A quick tunnel comes with no uptime guarantee and allows at most 200
  requests in flight.
* The address changes on every start, so show `/share` rather than a QR code
  printed in advance.
* Visitors' registrations are real ledger entries, like every other.
* Some networks block Cloudflare's tunnel port (7844); a phone hotspot works.
* Cloudflare gives up on an answer after 100 seconds, which a long video's
  analysis can exceed.

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
