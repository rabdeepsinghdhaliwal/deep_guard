// Deep-Guard ledger — every check redone inside the browser.
//
// The server says "block #3 is fine". This file lets the page NOT take its
// word for it: it rebuilds the same canonical JSON, the same SHA-256 hashes,
// the same Merkle tree and checks the same Ed25519 signatures with the
// browser's own Web Crypto, byte for byte the way ledger_core.py does.
// If the two ever disagree, the page shows it.
"use strict";

const DG = (() => {
  const enc = new TextEncoder();
  const HEADER_FIELDS = ["index", "timestamp", "previous_hash", "merkle_root", "record_count", "signer_key_id"];
  const GENESIS_PREVIOUS_HASH = "0".repeat(64);

  // Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  function escapeAscii(s) {
    return s.replace(/[\u007f-￿]/g, (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
  }
  function canonical(value) {
    if (value === null || typeof value !== "object") {
      if (typeof value === "number" && !Number.isInteger(value)) throw new TypeError("ledger records hold no floats");
      return escapeAscii(JSON.stringify(value));
    }
    if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
    return "{" + Object.keys(value).sort().map((k) => escapeAscii(JSON.stringify(k)) + ":" + canonical(value[k])).join(",") + "}";
  }

  function hexToBytes(hex) {
    const clean = String(hex || "");
    const out = new Uint8Array(Math.floor(clean.length / 2));
    for (let i = 0; i < out.length; i++) out[i] = parseInt(clean.substr(i * 2, 2), 16);
    return out;
  }
  function bytesToHex(bytes) {
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }
  function concat(...parts) {
    const total = parts.reduce((n, p) => n + p.length, 0);
    const out = new Uint8Array(total);
    let at = 0;
    for (const p of parts) { out.set(p, at); at += p.length; }
    return out;
  }
  async function sha256(bytes) {
    return bytesToHex(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
  }

  const leafHash = (record) => sha256(concat(new Uint8Array([0]), enc.encode(canonical(record))));
  const nodeHash = (l, r) => sha256(concat(new Uint8Array([1]), hexToBytes(l), hexToBytes(r)));

  async function merkleLevels(records) {
    if (!records || !records.length) return [[await sha256(new Uint8Array())]];
    let level = await Promise.all(records.map(leafHash));
    const levels = [level];
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i < level.length; i += 2) {
        next.push(i + 1 < level.length ? await nodeHash(level[i], level[i + 1]) : level[i]);
      }
      level = next;
      levels.push(level);
    }
    return levels;
  }
  async function merkleRoot(records) {
    const levels = await merkleLevels(records);
    return levels[levels.length - 1][0];
  }
  async function verifyProof(leaf, proof, root) {
    let current = leaf;
    for (const step of proof || []) {
      current = step.side === "left" ? await nodeHash(step.hash, current) : await nodeHash(current, step.hash);
    }
    return current === root;
  }
  function header(block) {
    const h = {};
    for (const f of HEADER_FIELDS) h[f] = block[f] === undefined ? null : block[f];
    return h;
  }
  const blockHash = (block) => sha256(enc.encode(canonical(header(block))));

  // ---------------------------------------------------------------- Ed25519
  const keyCache = new Map();
  async function importKey(pem) {
    if (keyCache.has(pem)) return keyCache.get(pem);
    const b64 = String(pem).replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
    const der = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const key = await crypto.subtle.importKey("spki", der, { name: "Ed25519" }, true, ["verify"]);
    keyCache.set(pem, key);
    return key;
  }
  // true / false, or null when this browser cannot check Ed25519 at all
  async function verifySig(pem, sigHex, dataBytes) {
    if (!(await ed25519Supported())) return null;
    let key;
    try { key = await importKey(pem); } catch (e) { return false; }
    try { return await crypto.subtle.verify({ name: "Ed25519" }, key, hexToBytes(sigHex), dataBytes); }
    catch (e) { return false; }
  }
  async function keyId(pem) {
    const key = await importKey(pem);
    const raw = new Uint8Array(await crypto.subtle.exportKey("raw", key));
    return (await sha256(raw)).slice(0, 16);
  }
  let supported = null;
  async function ed25519Supported() {
    if (supported !== null) return supported;
    try {
      // a throwaway key, just to ask "does this browser know Ed25519?"
      await crypto.subtle.generateKey({ name: "Ed25519" }, false, ["sign", "verify"]);
      supported = true;
    } catch (e) { supported = false; }
    return supported;
  }
  const witnessStatement = (index, hash) => enc.encode(`deepguard-witness-v1|${index}|${hash}`);

  function unsignedRecord(record) {
    const copy = { ...record };
    delete copy.creator_signature;
    return copy;
  }
  async function creatorSignatureOk(record) {
    if (!record.creator_public_key) return null;
    return verifySig(record.creator_public_key, record.creator_signature || "", enc.encode(canonical(unsignedRecord(record))));
  }

  const check = (name, ok, detail = "") => ({ name, ok, detail });
  const short = (h) => (h ? String(h).slice(0, 16) + "…" : String(h));

  // The same checks, with the same names, as ledger_core.check_block
  async function checkBlock(block, position, previous, pinnedPem) {
    if (block._unreadable) {
      return { index: position, ok: false, checks: [check("Block can be read", false, `Line ${block._unreadable} of chain.jsonl is not valid JSON any more`)] };
    }
    const checks = [];
    const records = block.records || [];
    checks.push(check("Position number is correct", block.index === position, `says #${block.index}, sits at position ${position}`));
    const expectedPrev = previous ? previous.block_hash : GENESIS_PREVIOUS_HASH;
    checks.push(check("Points to the previous block", block.previous_hash === expectedPrev,
      block.previous_hash === expectedPrev ? "" : `stores ${short(block.previous_hash)}, the block before hashes to ${short(expectedPrev)}`));
    let root;
    try {
      root = await merkleRoot(records);
      const ok = root === block.merkle_root && block.record_count === records.length;
      checks.push(check("Records still match the Merkle root", ok,
        ok ? "" : root !== block.merkle_root ? `records now hash to ${short(root)}, block says ${short(block.merkle_root)}` : `record_count says ${block.record_count}, block holds ${records.length}`));
    } catch (e) { checks.push(check("Records still match the Merkle root", false, String(e.message || e))); }
    try {
      const h = await blockHash(block);
      checks.push(check("Block hash matches its contents", h === block.block_hash, h === block.block_hash ? "" : `contents hash to ${short(h)}, block says ${short(block.block_hash)}`));
    } catch (e) { checks.push(check("Block hash matches its contents", false, String(e.message || e))); }

    let keyOk = false;
    try { keyOk = block.signer_public_key === pinnedPem && block.signer_key_id === (await keyId(pinnedPem)); } catch (e) { keyOk = null; }
    let sig = keyOk === false ? false : await verifySig(pinnedPem, block.signature || "", hexToBytes(block.block_hash || ""));
    if (keyOk === null || sig === null) checks.push({ name: "Signed by Deep-Guard's key", ok: null, detail: "this browser cannot check Ed25519 signatures — the server's check is shown instead" });
    else checks.push(check("Signed by Deep-Guard's key", keyOk && sig, keyOk ? (sig ? "" : "signature does not match this block hash") : "signed by a different key"));

    const bad = [];
    let unknown = false;
    for (let i = 0; i < records.length; i++) {
      const r = await creatorSignatureOk(records[i]);
      if (r === false) bad.push(i);
      if (r === null && records[i].creator_public_key) unknown = true;
    }
    if (unknown && !bad.length) checks.push({ name: "Every creator signature is valid", ok: null, detail: "this browser cannot check Ed25519 signatures" });
    else checks.push(check("Every creator signature is valid", !bad.length, bad.length ? `record(s) at position ${bad.join(", ")} fail` : ""));

    if (previous && previous.timestamp && block.timestamp) {
      checks.push(check("Time only moves forward", block.timestamp >= previous.timestamp, block.timestamp >= previous.timestamp ? "" : "this block claims to be older than the one before it"));
    }
    return { index: position, block_hash: block.block_hash, ok: checks.every((c) => c.ok !== false), checks };
  }

  async function checkChain(blocks, pinnedPem) {
    const results = [];
    let previous = null;
    for (let i = 0; i < blocks.length; i++) {
      results.push(await checkBlock(blocks[i], i, previous, pinnedPem));
      previous = blocks[i]._unreadable ? { block_hash: null } : blocks[i];
    }
    const bad = results.filter((r) => !r.ok).map((r) => r.index);
    return { ok: !bad.length && blocks.length > 0, blocks: results, first_bad_index: bad.length ? bad[0] : null, bad_indexes: bad };
  }

  // Same checks as ledger_core.verify_receipt
  async function verifyReceipt(receipt, pinnedPem) {
    const checks = [];
    const record = receipt.record, hdr = receipt.block_header;
    const leaf = await leafHash(record);
    checks.push(check("Record hashes to the leaf in the receipt", leaf === receipt.leaf_hash));
    checks.push(check("Merkle path leads to the block's root", await verifyProof(leaf, receipt.merkle_proof, hdr.merkle_root)));
    checks.push(check("Block hash matches the header", (await sha256(enc.encode(canonical(hdr)))) === receipt.block_hash));
    let kid = null;
    try { kid = await keyId(pinnedPem); } catch (e) { /* unsupported */ }
    checks.push(kid === null ? { name: "Header names the expected signing key", ok: null, detail: "browser cannot read Ed25519 keys" } : check("Header names the expected signing key", hdr.signer_key_id === kid));
    const sig = await verifySig(pinnedPem, receipt.signature, hexToBytes(receipt.block_hash));
    checks.push(sig === null ? { name: "Deep-Guard's signature is valid", ok: null, detail: "browser cannot check Ed25519" } : check("Deep-Guard's signature is valid", sig));
    const creator = await creatorSignatureOk(record);
    if (creator !== null || record.creator_public_key) checks.push(creator === null ? { name: "Creator's signature is valid", ok: null, detail: "browser cannot check Ed25519" } : check("Creator's signature is valid", creator));
    for (const c of receipt.witness_cosignatures || []) {
      const ok = await verifySig(c.witness_public_key, c.signature, witnessStatement(hdr.index, receipt.block_hash));
      checks.push(ok === null ? { name: `${c.witness} co-signed this block`, ok: null, detail: "browser cannot check Ed25519" } : check(`${c.witness} co-signed this block`, ok));
    }
    return { ok: checks.every((c) => c.ok !== false), checks, leaf };
  }

  // ------------------------------------------------------- Bitcoin anchors
  // Replays an OpenTimestamps path (anchoring.attestation_paths) on a starting
  // hash, the same operations the proof file holds. Returns the final hex, or
  // null when the path needs an operation Web Crypto doesn't have
  // (ripemd160 or keccak256, found in older proofs): the server's check is
  // shown for those instead.
  async function replayOts(startHex, ops) {
    let msg = hexToBytes(startHex);
    for (const step of ops || []) {
      if (step.op === "append") msg = concat(msg, hexToBytes(step.arg));
      else if (step.op === "prepend") msg = concat(hexToBytes(step.arg), msg);
      else if (step.op === "sha256") msg = new Uint8Array(await crypto.subtle.digest("SHA-256", msg));
      else return null;
    }
    return bytesToHex(msg);
  }
  // Explorers print a Merkle root byte-reversed; a proof ends on it as stored in the block header.
  const reverseHex = (hex) => (String(hex).match(/../g) || []).reverse().join("");

  async function getWithTimeout(url, kind) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 12000);
    try {
      const r = await fetch(url, { signal: ctrl.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return kind === "json" ? await r.json() : (await r.text()).trim();
    } finally { clearTimeout(t); }
  }
  // Bitcoin block `height`, fetched by this page from two explorers run by
  // different companies (not through the Deep-Guard server). They must agree.
  async function bitcoinBlock(height) {
    const sources = [
      ["blockstream.info", async () => {
        const id = await getWithTimeout(`https://blockstream.info/api/block-height/${height}`, "text");
        const b = await getWithTimeout(`https://blockstream.info/api/block/${id}`, "json");
        return { block_hash: b.id, merkle_root: b.merkle_root, time: b.timestamp };
      }],
      ["blockcypher.com", async () => {
        const b = await getWithTimeout(`https://api.blockcypher.com/v1/btc/main/blocks/${height}?txstart=0&limit=1`, "json");
        return { block_hash: b.hash, merkle_root: b.mrkl_root, time: Math.floor(new Date(b.time).getTime() / 1000) };
      }],
    ];
    const got = [];
    for (const [name, lookup] of sources) {
      try { got.push([name, await lookup()]); } catch (e) { /* unreachable: try the next */ }
    }
    if (!got.length) throw new Error("no block explorer answered this browser (offline?)");
    for (const [name, b] of got.slice(1)) {
      if (b.merkle_root !== got[0][1].merkle_root) throw new Error(`the explorers disagree (${got[0][0]} vs ${name})`);
    }
    return { ...got[0][1], height, sources: got.map(([name]) => name) };
  }

  return {
    canonical, sha256, leafHash, nodeHash, merkleLevels, merkleRoot, verifyProof, blockHash, header,
    verifySig, keyId, ed25519Supported, witnessStatement, creatorSignatureOk, checkBlock, checkChain,
    verifyReceipt, hexToBytes, short, replayOts, reverseHex, bitcoinBlock,
  };
})();

// Small shared UI helpers for both ledger pages
const DGUI = {
  esc(s) {
    return String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  },
  time(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  },
  checksHtml(checks) {
    return '<ul class="dgl-checks">' + checks.map((c) => {
      const cls = c.ok === true ? "is-ok" : c.ok === false ? "is-bad" : "is-skip";
      const tick = c.ok === true ? "✓" : c.ok === false ? "✗" : "?";
      return `<li class="${cls}"><span class="dgl-tick">${tick}</span><span>${DGUI.esc(c.name)}${c.detail ? `<small>${DGUI.esc(c.detail)}</small>` : ""}</span></li>`;
    }).join("") + "</ul>";
  },
  async errorText(response) {
    try {
      const body = await response.json();
      const d = body.detail;
      if (typeof d === "string") return d;
      if (d && d.message) return d.message;
      return JSON.stringify(d || body);
    } catch (e) { return `Request failed (HTTP ${response.status}).`; }
  },
  originLabel(o) {
    return { camera: "Camera original", edited: "Edited by its creator", "ai-generated": "AI-generated (declared)" }[o] || o;
  },

  // The chain of trust for one record, every link recomputed in the browser:
  // record -> leaf hash -> Merkle path -> root -> block hash -> signature -> witnesses
  async proofSteps(bundle, pinnedPem) {
    const steps = [];
    const add = (ok, title, detail) => steps.push({ ok, title, detail });
    const rec = bundle.record;
    const creator = await DG.creatorSignatureOk(rec);
    if (creator !== null || rec.creator_public_key) {
      add(creator, "Creator signed the record",
        `${rec.creator_handle || rec.creator_id} (key ${rec.creator_key_id || "?"}) signed every field of this record`);
    } else {
      add(true, "Written by Deep-Guard", rec.resolved_by || "a service record, covered by the block signature");
    }
    if (!bundle.sealed && bundle.sealed !== undefined) {
      add(null, "Not sealed yet", "This record is in the pending pool: no block, Merkle proof or block signature yet.");
      return steps;
    }
    const leaf = await DG.leafHash(rec);
    add(leaf === bundle.leaf_hash, "Record → leaf hash", `SHA-256(0x00 + record) = ${DG.short(leaf)}`);
    const pathOk = await DG.verifyProof(leaf, bundle.merkle_proof, bundle.block_header.merkle_root);
    const steps_n = bundle.merkle_proof.length;
    add(pathOk, "Merkle path → block's root", steps_n === 0
      ? `The only record in its block, so its leaf hash is the root itself (${DG.short(bundle.block_header.merkle_root)})`
      : `${steps_n} sibling hash${steps_n === 1 ? " leads" : "es lead"} to root ${DG.short(bundle.block_header.merkle_root)}`);
    const h = await DG.sha256(new TextEncoder().encode(DG.canonical(bundle.block_header)));
    add(h === bundle.block_hash, `Header → block #${bundle.block_header.index} hash`, `SHA-256(header) = ${DG.short(h)}`);
    let kid = null;
    try { kid = await DG.keyId(pinnedPem); } catch (e) { /* unsupported */ }
    const sig = await DG.verifySig(pinnedPem, bundle.signature, DG.hexToBytes(bundle.block_hash));
    add(sig === null || kid === null ? null : sig && kid === bundle.block_header.signer_key_id, "Deep-Guard's signature",
      sig === null ? "This browser cannot check Ed25519 signatures (the server checked it)." : `Ed25519 signature by key ${kid}`);
    const cosigs = bundle.witness_cosignatures || [];
    if (!cosigs.length) add(null, "Witness co-signatures", "None yet — the witnesses were offline when this block was sealed. Sync them from the explorer.");
    for (const c of cosigs) {
      const ok = await DG.verifySig(c.witness_public_key, c.signature, DG.witnessStatement(bundle.block_header.index, bundle.block_hash));
      add(ok, `${c.witness} co-signed`, `"I saw block #${bundle.block_header.index} with this hash" — key ${c.witness_key_id}`);
    }
    return steps;
  },
  proofHtml(steps) {
    return '<div class="dgl-proof">' + steps.map((s) => {
      const cls = s.ok === true ? "is-ok" : s.ok === false ? "is-bad" : "is-skip";
      const icon = s.ok === true ? "✓" : s.ok === false ? "✗" : "…";
      return `<div class="dgl-proof__step ${cls}"><span class="dgl-proof__icon">${icon}</span><div><div class="dgl-proof__title">${DGUI.esc(s.title)}</div><div class="dgl-proof__detail">${DGUI.esc(s.detail)}</div></div></div>`;
    }).join("") + "</div>";
  },
};
