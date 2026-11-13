# Stage 4 — durable provenance: the build plan

Five additions, built one at a time on the branch `stage4-durable-provenance`.
Each phase ends working and tested before the next one starts. Number 4 from
the ideas list (witnesses on three laptops) waits until the team is together;
7 and 8 are out of scope for now.

## Where they plug in

Stage 3 was built beside the bouncer without changing a single bouncer file:
`deepguard-ledger/ledger_app.py` imports the bouncer's app and adds to it, and
`static/bridge.js` adds cards to the old pages at serve time. Stage 4 keeps
that rule. New server code goes into `deepguard-ledger/`, the browser
extension into its own folder `deepguard-lens/`, and the bouncer (and its 119
tests) stays untouched.

Step 0 already defined the answer format for exactly these layers
(`deepguard-bouncer/app/evidence.py`): layer 0 Content Credentials, layer 1
Watermarks, layer 2 Detectors, layer 3 Known-content match. Each new check
reports in that shape.

## The order, and why

| # | Phase | Why this position |
|---|---|---|
| 1 | Anchor the ledger to Bitcoin (OpenTimestamps) | Independent of everything else, and Bitcoin takes a few hours to confirm. Building it first lets the first anchor confirm while the rest is built. |
| 2 | Hidden watermark ID at registration (TrustMark) | The foundation of "durable" credentials: phase 3 writes this ID into the manifest. |
| 3 | Read and write the CR badge (C2PA) | Reads any file's Content Credentials; writes Deep-Guard's own, pointing at the watermark ID and the ledger record. |
| 4 | Deep-Guard Lens browser extension | A client of the finished APIs (ledger + C2PA + watermark + detector), so it is built once against stable endpoints. |
| 5 | Live public link with a QR code | Wraps everything. It needs safety limits across all endpoints, including the new ones, so it comes last. |
| — | Final pass | Docs, demo runbook, a fresh-clone dry run of the whole thing. |

## Phase 1 — Anchor the ledger to Bitcoin

**What it proves.** Today the ledger plus its witnesses could, in theory, agree
to backdate history together. Once a block's hash is inside a Bitcoin block,
nobody can: not Deep-Guard, not the witnesses, not us.

**How.**
- Anchor the newest block's hash. Every block commits to all the blocks before
  it through `previous_hash`, so one anchor covers the whole history up to it.
- `block_hash = SHA-256(canonical header bytes)`, so the canonical header is
  itself the "file" OpenTimestamps timestamps. Saving it as
  `block-N.header.json` next to `block-N.header.json.ots` means anyone can drop
  the two files on opentimestamps.org and see the Bitcoin attestation, with no
  Deep-Guard code at all.
- Submit to several public OpenTimestamps calendars (only the 32-byte hash
  leaves the machine). The proof is "pending" until the calendar's aggregate is
  mined, usually a few hours; then "upgrade" fetches the complete path into a
  Bitcoin block.
- Verify our own way: replay the proof's operations, then compare the result
  with that Bitcoin block's Merkle root, fetched from two independent block
  explorers that must agree. The explorer page shows the path; a new
  `verify_ledger.py anchors` command checks it offline.
- Anchors live beside the chain (`ledger_data/anchors/`), like witness
  co-signatures: they are proofs *about* blocks, so they cannot live inside them.

**Done when:** an "Anchor to Bitcoin" button works, a pending anchor upgrades
to a confirmed one, the explorer shows "block #N is in Bitcoin block #H, mined
<date>", and the check passes in the server, in Python offline and on
opentimestamps.org. Tests use OpenTimestamps' own published example proof so
they run offline.

## Phase 2 — Hidden watermark ID

**What it proves.** Metadata gets stripped on upload. An ID hidden in the
pixels survives WhatsApp, screenshots and crops, so a copy can still be traced
to its record.

**How.**
- At registration (images), embed a random 61-bit ID with TrustMark (Adobe's
  open watermark; variant Q, BCH_5 error correction, which fixes up to 5 wrong
  bits) into a copy of the image. The creator downloads this credentialed copy
  and shares it.
- The record stores the ID and the fingerprints of both the original and the
  marked copy, signed by the creator like everything else.
- Verifying reads the ID first. It is accepted only when the content also
  matches (SSCD), because an ID can be found by chance or copied onto another
  picture (October measurement: a false ID 1 time in 120 unmarked pictures).
- Measured beforehand on 60 exam images: the ID survived 60/60 WhatsApp-style
  copies and 57/60 crop+WhatsApp copies; the marked copy looks identical
  (42 dB PSNR median). Phase 2 re-measures this on the final code.

**Done when:** register → download marked copy → re-save, resize, crop it →
verify says "hidden ID found in the pixels → Family photo by Rabdeep", and the
numbers above are reproduced by a test script.

## Phase 3 — The CR badge (C2PA Content Credentials)

**Read.** Every analysed or verified file is checked for an embedded C2PA
manifest: who signed it, with what tool, the edit history, whether it declares
AI generation, and whether the signature and the file still check out. The
analysis page and the ledger page get a "Content Credentials" card with the CR
mark. Tested on the 27 official C2PA test files already in `~/deepguard-data`.

**Write.** The credentialed copy from phase 2 also gets a manifest signed by
Deep-Guard: the declared origin (camera / edited / AI-generated as an IPTC
digital source type), the creator, a `c2pa.soft-binding` assertion carrying the
watermark ID (`com.adobe.trustmark.Q`, the format TrustMark's own C2PA example
uses), and the ledger record ID. This is the "three pillars" design of durable
Content Credentials: manifest + watermark + fingerprint.

**Honest limit.** Signing needs an X.509 certificate. We create our own small
certificate authority ("Deep-Guard Demo Root"); our own pages trust it, while
outside validators such as contentcredentials.org/verify will show the
manifest but call the signer unknown. Certificates on the official C2PA trust
list require the conformance programme.

## Phase 4 — Deep-Guard Lens (browser extension)

A Manifest V3 extension for Edge and Chrome:
- Right-click any image on any website → "Check with Deep-Guard". The
  extension sends the image to your Deep-Guard server (local or the public
  link) and pins a badge on the picture: ledger verdict, CR badge, watermark
  ID, detector reading.
- The toolbar popup sets the server address, shows whether it is reachable, and
  can scan the visible images on a page.
- Installed with "Load unpacked"; nothing is published to a store.

## Phase 5 — Live public link

- `python run_deepguard.py --public` starts a Cloudflare quick tunnel (free, no
  account) and prints an https address plus a QR code; a `/share` page shows
  the QR full-screen for the projector.
- Public mode turns on safety limits: smaller uploads, a per-visitor request
  rate, one heavy job at a time, admin actions switched off, and local file
  paths hidden.
- The witnesses get their own tunnels so phones can still check them directly.
- Needs one download: `cloudflared.exe` from Cloudflare's GitHub releases.

## Final pass

README sections, the ledger README, an updated demo runbook (new acts for
each feature), and a dry run of the whole thing from a fresh clone.
