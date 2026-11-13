// Deep-Guard — /explorer: the whole chain, drawn and re-checked in the browser.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = DGUI.esc;
const S = { chain: null, check: null, checkSource: null, selectedBlock: null, selectedRecord: null, creators: [], witnessView: [], anchors: null };
const anchorOf = (index) => (S.anchors ? S.anchors.anchors.find((a) => a.block_index === index) : null);

const blocks = () => (S.chain ? S.chain.blocks : []);
const pinned = () => S.chain.service_public_key;
const shortName = (name) => String(name).split(" ").pop();

function setStatus(html) { $("statusLine").innerHTML = html; }

// --------------------------------------------------------------- loading --

async function load({ recheck = true, newIndex = null } = {}) {
  const [chain, creators, anchors] = await Promise.all([
    fetch("/api/ledger/chain").then((r) => r.json()),
    fetch("/api/ledger/creators").then((r) => r.json()),
    fetch("/api/ledger/anchors").then((r) => r.json()).catch(() => null),
  ]);
  S.chain = chain;
  S.creators = creators.creators;
  S.anchors = anchors;
  renderStats();
  renderChain(newIndex);
  fillTamperBlocks();
  renderKeys();
  renderAnchors();
  if (recheck) await browserCheck();
  loadWitnesses();
  if (S.selectedBlock !== null && S.selectedBlock < blocks().length) await selectBlock(S.selectedBlock, { scroll: false });
}

// ----------------------------------------------------------------- stats --

function allRecords() {
  const out = [];
  blocks().forEach((b) => (b.records || []).forEach((r, pos) => out.push({ record: r, block: b.index, position: pos })));
  (S.chain.pending || []).forEach((r, pos) => out.push({ record: r, block: null, position: pos }));
  return out;
}

function renderStats() {
  const bl = blocks();
  const records = bl.reduce((n, b) => n + ((b.records || []).length), 0);
  let chainState = '<span class="dgl-muted">not checked</span>';
  if (S.check) {
    chainState = S.check.ok
      ? '<span style="color:var(--true)">intact</span>'
      : `<span style="color:var(--false)">broken at #${S.check.first_bad_index}</span>`;
  }
  const online = S.witnessView.filter((w) => w.online);
  const agree = online.filter((w) => w.agrees);
  const wit = S.witnessView.length ? `${agree.length}/${S.witnessView.length}` : "…";
  const anchored = S.anchors && S.anchors.anchors.length ? S.anchors.anchors[S.anchors.anchors.length - 1] : null;
  const anchorStat = !anchored ? '<span class="dgl-muted">none</span>'
    : `#${anchored.block_index} <span class="dgl-stat__tag ${anchored.status === "confirmed" ? "is-btc" : ""}">${anchored.status === "confirmed" ? "in Bitcoin" : esc(anchored.status)}</span>`;
  $("stats").innerHTML = [
    [bl.length, "blocks (incl. genesis #0)"],
    [records, "records sealed"],
    [(S.chain.pending || []).length, "records waiting (pending)"],
    [chainState, "chain status"],
    [wit, "witnesses agree"],
    [anchorStat, "newest Bitcoin anchor"],
  ].map(([n, l]) => `<div class="dgl-stat"><div class="dgl-stat__n">${n}</div><div class="dgl-stat__l">${l}</div></div>`).join("");
}

// ----------------------------------------------------------------- chain --

function blockLabel(b) {
  if (b._unreadable) return "Unreadable line";
  if (b.index === 0) return "Genesis";
  const recs = b.records || [];
  if (!recs.length) return "(empty)";
  const first = recs[0];
  const label = first.type === "registration" ? first.title : first.type[0].toUpperCase() + first.type.slice(1);
  return recs.length > 1 ? `${label} +${recs.length - 1}` : label;
}

function chainHtml(list, report, { mini = false, cosignatures = {}, witnesses = [], newIndex = null, forged = null } = {}) {
  const parts = [];
  list.forEach((b, i) => {
    if (i > 0) {
      const ok = !b._unreadable && !list[i - 1]._unreadable && b.previous_hash === list[i - 1].block_hash;
      parts.push(`<div class="dgl-link ${ok ? "is-ok" : "is-bad"}" title="${ok ? "Stores the previous block's hash" : "Does NOT match the previous block's hash"}">
        <span class="dgl-link__txt">${ok ? "prev ✓" : "broken ✗"}</span><span class="dgl-link__line"></span>
        <span class="dgl-link__txt">${esc(String(b.previous_hash || "").slice(0, 8))}</span></div>`);
    }
    const r = report && report.blocks[i];
    const isForged = forged && forged.has(i);
    const state = r ? (!r.ok ? "is-bad" : isForged ? "is-forged" : "is-ok") : "";
    const cos = cosignatures[String(b.index)] || [];
    const wit = mini ? "" : witnesses.map((w) => {
      const has = cos.some((c) => c.witness === w.name);
      return `<span class="${has ? "is-ok" : ""}" title="${esc(w.name)} ${has ? "co-signed" : "has not co-signed"}">${esc(shortName(w.name))}${has ? " ✓" : ""}</span>`;
    }).join("");
    const anchor = mini ? null : anchorOf(b.index);
    const btc = !anchor ? "" : anchor.status === "confirmed"
      ? `<span class="dgl-pill dgl-pill--btc" style="align-self:flex-start" title="Anchored in Bitcoin block #${esc(anchor.bitcoin && anchor.bitcoin.height)}">₿ in Bitcoin</span>`
      : `<span class="dgl-pill dgl-pill--warn" style="align-self:flex-start" title="Sent to the OpenTimestamps calendars; waiting for Bitcoin">₿ ${esc(anchor.status)}</span>`;
    parts.push(`<button type="button" class="dgl-block ${state} ${i === newIndex ? "is-new" : ""}" data-index="${i}" ${mini ? 'tabindex="-1"' : ""}
        aria-label="Block ${i}${r ? (r.ok ? ", passes all checks" : ", fails a check") : ""}">
      <span class="dgl-block__num">#${esc(b._unreadable ? "?" : b.index)}</span>
      <span class="dgl-block__label">${esc(blockLabel(b))}</span>
      <span class="dgl-block__hash">${esc(String(b.block_hash || "").slice(0, 12))}…</span>
      <span class="dgl-block__meta">${(b.records || []).length} record${(b.records || []).length === 1 ? "" : "s"} · ${esc(DGUI.time(b.timestamp))}</span>
      ${r ? `<span class="dgl-pill ${r.ok ? "dgl-pill--ok" : "dgl-pill--bad"}" style="align-self:flex-start">${r.ok ? "✓ checks pass" : "✗ " + esc((r.checks.find((c) => c.ok === false) || {}).name || "fails")}</span>` : ""}
      ${isForged ? '<span class="dgl-pill dgl-pill--warn" style="align-self:flex-start">≠ witnesses’ copy</span>' : ""}
      ${btc}
      ${wit ? `<span class="dgl-block__wit">${wit}</span>` : ""}
    </button>`);
  });
  return parts.join("");
}

function renderChain(newIndex = null) {
  const bl = blocks();
  let html = chainHtml(bl, S.check, { cosignatures: S.chain.cosignatures, witnesses: S.chain.witnesses, newIndex });
  const pending = S.chain.pending || [];
  if (pending.length) {
    html += `<div class="dgl-link"><span class="dgl-link__txt">next</span><span class="dgl-link__line"></span><span class="dgl-link__txt">&nbsp;</span></div>
      <div class="dgl-block is-pending"><span class="dgl-block__num">…</span><span class="dgl-block__label">Pending pool</span>
      <span class="dgl-block__meta">${pending.length} record${pending.length === 1 ? "" : "s"} waiting for a block</span>
      <span class="dgl-block__wit"><button type="button" class="dgl-btn dgl-btn--small" data-seal>Seal now</button></span></div>`;
  }
  $("chain").innerHTML = html;
  $("chain").querySelectorAll(".dgl-block[data-index]").forEach((el) => {
    el.addEventListener("click", () => { location.hash = `block=${el.dataset.index}`; });
    if (Number(el.dataset.index) === S.selectedBlock) el.classList.add("is-selected");
  });
  const sealBtn = $("chain").querySelector("[data-seal]");
  if (sealBtn) sealBtn.addEventListener("click", seal);
  if (newIndex !== null) $("chainWrap").scrollLeft = $("chainWrap").scrollWidth;
}

// ---------------------------------------------------------------- checks --

async function browserCheck() {
  const n = blocks().length;
  setStatus(`Recomputing ${n} block${n === 1 ? "" : "s"} in your browser…`);
  const supported = await DG.ed25519Supported();
  S.check = await DG.checkChain(blocks(), pinned());
  S.checkSource = "browser";
  renderStats();
  renderChain();
  const sigNote = supported ? "" : " (this browser cannot check Ed25519 signatures; use “Ask the server to verify” for those)";
  if (S.check.ok) {
    setStatus(`<span class="dgl-pill dgl-pill--ok">✓ intact</span> All ${n} blocks pass every check — recomputed in your browser: hashes, Merkle roots, links and signatures${sigNote}.`);
  } else {
    const bad = S.check.blocks[S.check.first_bad_index];
    const failed = bad.checks.find((c) => c.ok === false);
    setStatus(`<span class="dgl-pill dgl-pill--bad">✗ tampered</span> Block #${S.check.first_bad_index} fails “${esc(failed.name)}”${failed.detail ? ` — ${esc(failed.detail)}` : ""}. ${S.check.bad_indexes.length > 1 ? `Blocks failing: ${S.check.bad_indexes.map((i) => "#" + i).join(", ")}.` : ""}`);
  }
}

async function serverCheck() {
  setStatus("Asking the server to verify the whole chain…");
  const report = await (await fetch("/api/ledger/verify-chain")).json();
  S.check = report;
  S.checkSource = "server";
  renderStats();
  renderChain();
  const w = (report.witnesses || []).map((x) => !x.online ? `${esc(x.name)}: offline` : x.first_mismatch !== null && x.first_mismatch !== undefined ? `${esc(x.name)}: DISAGREES at #${x.first_mismatch}` : `${esc(x.name)}: agrees${x.missing_blocks ? ` (missing ${x.missing_blocks} newest)` : ""}`).join(" · ");
  setStatus(report.ok
    ? `<span class="dgl-pill dgl-pill--ok">✓ server: intact</span> ${report.block_count} blocks, ${report.record_count} records checked. ${w}`
    : `<span class="dgl-pill dgl-pill--bad">✗ server: broken</span> first bad block #${report.first_bad_index}. ${w}`);
  if (S.selectedBlock !== null) selectBlock(S.selectedBlock, { scroll: false });
}

// ---------------------------------------------------------------- blocks --

function treeSvg(levels, labels, leafIndex = null) {
  const n = levels[0].length;
  const H = levels.length;
  const nodeW = 96, nodeH = 24, gapX = 16, levelGap = 60;
  const width = Math.max(n * (nodeW + gapX) + 24, 380);
  const height = H * levelGap + 46;
  const xs = [];
  const span = n * (nodeW + gapX) - gapX;
  const start = (width - span) / 2 + nodeW / 2;
  xs[0] = levels[0].map((_, i) => start + i * (nodeW + gapX));
  for (let L = 1; L < H; L++) {
    xs[L] = levels[L].map((_, j) => {
      const a = xs[L - 1][2 * j], b = xs[L - 1][2 * j + 1];
      return b === undefined ? a : (a + b) / 2;
    });
  }
  const yOf = (L) => 16 + (H - 1 - L) * levelGap;
  const onPath = (L, i) => leafIndex !== null && i === (leafIndex >> L);
  const isSibling = (L, i) => leafIndex !== null && L < H - 1 && i === ((leafIndex >> L) ^ 1);
  let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="Merkle tree of this block">`;
  for (let L = 1; L < H; L++) {
    levels[L - 1].forEach((_, i) => {
      const p = i >> 1;
      const path = onPath(L - 1, i) && onPath(L, p);
      svg += `<path class="edge ${path ? "is-path" : ""}" d="M${xs[L - 1][i]} ${yOf(L - 1)} L${xs[L][p]} ${yOf(L) + nodeH}"/>`;
    });
  }
  levels.forEach((level, L) => level.forEach((h, i) => {
    const cls = ["node", L === H - 1 ? "is-root" : "", L === 0 && labels.length ? "is-leaf" : "", onPath(L, i) ? "is-path" : "", isSibling(L, i) ? "is-sibling" : ""].join(" ");
    svg += `<g class="${cls}" ${L === 0 && labels.length ? `data-leaf="${i}"` : ""}><title>${esc(h)}</title>
      <rect x="${xs[L][i] - nodeW / 2}" y="${yOf(L)}" width="${nodeW}" height="${nodeH}" rx="3"/>
      <text x="${xs[L][i]}" y="${yOf(L) + 16}" text-anchor="middle">${esc(h.slice(0, 10))}…</text></g>`;
  }));
  labels.forEach((label, i) => {
    svg += `<text class="leaf-label" x="${xs[0][i]}" y="${yOf(0) + nodeH + 16}" text-anchor="middle">${esc(label.length > 15 ? label.slice(0, 14) + "…" : label)}</text>`;
  });
  svg += `<text class="lvl-label" x="8" y="${yOf(H - 1) + 16}">root</text>`;
  if (H > 1) svg += `<text class="lvl-label" x="8" y="${yOf(0) + 16}">leaves</text>`;
  return svg + "</svg>";
}

async function selectBlock(index, { scroll = true, leafIndex = null } = {}) {
  const b = blocks()[index];
  if (!b) return;
  S.selectedBlock = index;
  document.querySelectorAll("#chain .dgl-block").forEach((el) => el.classList.toggle("is-selected", Number(el.dataset.index) === index));
  const panel = $("blockPanel");
  if (b._unreadable) {
    panel.innerHTML = `<div class="dgl-card"><h2 class="dgl-section__title">Line ${b._unreadable} cannot be read</h2>
      <p>This line of chain.jsonl is no longer valid JSON — it was edited by hand. Raw start:</p><pre class="dgl-json">${esc(b.raw)}</pre></div>`;
    panel.hidden = false;
    return;
  }
  const records = b.records || [];
  let levels;
  try { levels = await DG.merkleLevels(records); } catch (e) { levels = [[b.merkle_root || "?"]]; }
  const r = S.check ? S.check.blocks[index] : null;
  const prev = index > 0 ? blocks()[index - 1] : null;
  const cos = (S.chain.cosignatures || {})[String(index)] || [];
  const cosChecks = [];
  for (const c of cos) {
    const ok = await DG.verifySig(c.witness_public_key, c.signature, DG.witnessStatement(b.index, b.block_hash));
    cosChecks.push({ name: `${c.witness} co-signed “I saw block #${b.index} with this hash”`, ok, detail: `witness key ${c.witness_key_id}` });
  }
  if (!cos.length) cosChecks.push({ name: "No witness co-signatures yet", ok: null, detail: "The witnesses were offline when this block was sealed — press Sync witnesses." });

  const rows = [
    ["index", `#${b.index}`],
    ["timestamp", `${esc(b.timestamp)} <span class="dgl-muted">(${esc(DGUI.time(b.timestamp))})</span>`],
    ["previous_hash", `<span class="dgl-mono">${esc(b.previous_hash)}</span> ${index === 0 ? '<span class="dgl-pill">genesis: 64 zeros</span>' : (prev && prev.block_hash === b.previous_hash ? `<span class="dgl-pill dgl-pill--ok">= block #${index - 1}'s hash</span>` : '<span class="dgl-pill dgl-pill--bad">≠ previous block</span>')}`],
    ["merkle_root", `<span class="dgl-mono">${esc(b.merkle_root)}</span>`],
    ["record_count", String(b.record_count)],
    ["signer_key_id", `<span class="dgl-mono">${esc(b.signer_key_id)}</span> <span class="dgl-muted">(Deep-Guard)</span>`],
    ["block_hash", `<span class="dgl-mono">${esc(b.block_hash)}</span>`],
    ["signature", `<span class="dgl-mono">${esc(String(b.signature || "").slice(0, 32))}…</span>`],
  ];
  panel.innerHTML = `
    <div class="dgl-card">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap">
        <h2 class="dgl-section__title">Block #${b.index}${b.index === 0 ? " · genesis" : ""}</h2>
        ${r ? `<span class="dgl-pill ${r.ok ? "dgl-pill--ok" : "dgl-pill--bad"}">${r.ok ? "✓ passes every check" : "✗ fails a check"} · ${S.checkSource === "server" ? "checked by the server" : "checked in your browser"}</span>` : ""}
      </div>
      <p class="dgl-section__hint">The header (the first six fields) is hashed into <code>block_hash</code>, and Deep-Guard signs that hash.
        The records are tied in through <code>merkle_root</code>.</p>
      <div class="dgl-grid2">
        <div><dl class="dgl-fields">${rows.map(([k, v]) => `<dt class="dgl-mono">${k}</dt><dd>${v}</dd>`).join("")}</dl></div>
        <div>
          <h3 class="registry__panel-title">Checks</h3>
          ${r ? DGUI.checksHtml(r.checks) : '<p class="dgl-small dgl-muted">Press “Re-check in my browser”.</p>'}
          <h3 class="registry__panel-title" style="margin-top:16px">Witnesses</h3>
          ${DGUI.checksHtml(cosChecks)}
        </div>
      </div>
      <h3 class="registry__panel-title" style="margin-top:22px">Merkle tree (recomputed in your browser)</h3>
      <p class="dgl-small dgl-muted">Each leaf is one record's hash; pairs are hashed together up to the root.
        ${records.length ? "Click a leaf: the blue path is the record's route to the root, the amber boxes are its proof." : "An empty block's root is the hash of nothing."}</p>
      <div class="dgl-tree" id="treeBox">${treeSvg(levels, records.map((x) => x.title || x.type), leafIndex)}</div>
      <p class="dgl-small" style="margin-top:6px">${levels[levels.length - 1][0] === b.merkle_root
        ? `<span class="dgl-pill dgl-pill--ok">✓ root matches the header</span>`
        : `<span class="dgl-pill dgl-pill--bad">✗ recomputed root ${esc(DG.short(levels[levels.length - 1][0]))} ≠ header ${esc(DG.short(b.merkle_root))}</span>`}</p>
      ${records.length ? `<h3 class="registry__panel-title" style="margin-top:22px">Records in this block</h3>
      <table class="dgl-table"><tr><th>#</th><th>Type</th><th>What</th><th>Creator</th><th>Record ID</th></tr>
      ${records.map((x, i) => `<tr class="is-click ${S.selectedRecord === x.record_id ? "is-selected" : ""}" data-record="${esc(x.record_id)}"><td>${i}</td>
        <td><span class="dgl-type dgl-type--${esc(x.type)}">${esc(x.type)}</span></td>
        <td>${esc(x.title || x.reason || x.outcome || "")}</td><td>${esc(x.creator_handle || x.resolved_by || "")}</td>
        <td class="dgl-mono">${esc(String(x.record_id).slice(0, 12))}…</td></tr>`).join("")}</table>` : ""}
      <details class="dgl-details"><summary>Raw block (exactly as stored in chain.jsonl)</summary><pre class="dgl-json">${esc(JSON.stringify(b, null, 2))}</pre></details>
    </div>`;
  panel.hidden = false;
  panel.querySelectorAll("[data-record]").forEach((tr) => tr.addEventListener("click", () => { location.hash = `record=${tr.dataset.record}`; }));
  panel.querySelectorAll("[data-leaf]").forEach((g) => g.addEventListener("click", () => {
    const rec = records[Number(g.dataset.leaf)];
    if (rec) location.hash = `record=${rec.record_id}`;
  }));
  if (scroll) panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --------------------------------------------------------------- records --

function laterRecordsAbout(id) {
  return allRecords().filter((x) => x.record.target_record_id === id);
}

async function selectRecord(id) {
  const found = allRecords().find((x) => x.record.record_id === id);
  const panel = $("recordPanel");
  if (!found) {
    panel.innerHTML = `<div class="dgl-card"><p>No record with ID <span class="dgl-mono">${esc(id)}</span> on this ledger.</p></div>`;
    panel.hidden = false;
    return;
  }
  S.selectedRecord = id;
  const rec = found.record;
  if (found.block !== null) await selectBlock(found.block, { scroll: false, leafIndex: found.position });

  const rows = [["record_id", `<span class="dgl-mono">${esc(rec.record_id)}</span>`], ["type", `<span class="dgl-type dgl-type--${esc(rec.type)}">${esc(rec.type)}</span>`]];
  const add = (k, v) => { if (v !== undefined && v !== null && v !== "") rows.push([k, v]); };
  add("title", esc(rec.title));
  add("creator", rec.creator_handle ? `${esc(rec.creator_handle)} <span class="dgl-muted">· key ${esc(rec.creator_key_id)}</span>` : null);
  add("declared_origin", rec.declared_origin ? esc(DGUI.originLabel(rec.declared_origin)) : null);
  add("registered_at", rec.registered_at ? esc(DGUI.time(rec.registered_at)) : null);
  add("filed_at", rec.filed_at ? esc(DGUI.time(rec.filed_at)) : null);
  add("media", rec.media_type ? `${esc(rec.media_type)} · ${esc(rec.file_name)} · ${(rec.file_size_bytes / 1024).toFixed(1)} KB` : null);
  add("sha256_merkle_root", rec.sha256_merkle_root ? `<span class="dgl-mono">${esc(rec.sha256_merkle_root)}</span>` : null);
  add("phash", rec.phash ? `<span class="dgl-mono">${esc(rec.phash)}</span>` : null);
  add("phash_frames", rec.phash_frames ? `${rec.phash_frames.length} frame hashes` : null);
  add("content_vector_sha256", rec.content_vector_sha256 ? `<span class="dgl-mono">${esc(rec.content_vector_sha256)}</span>` : null);
  add("watermark", rec.watermark ? `hidden ID <span class="dgl-mono">${esc(rec.watermark.id)}</span> <span class="dgl-muted">· ${esc(rec.watermark.alg)}, ${esc(rec.watermark.schema)}</span>` : null);
  add("credentialed_copy", rec.credentialed_copy ? `${esc(rec.credentialed_copy.file_name)} · SHA-256 <span class="dgl-mono">${esc(DG.short(rec.credentialed_copy.sha256_merkle_root))}</span>` : null);
  add("detector_reading", rec.detector_reading ? `${esc(rec.detector_reading.ai_probability_percent)}% AI at registration${rec.detector_reading.calibrated_percent ? ` (calibrated ${esc(rec.detector_reading.calibrated_percent)}%)` : ""}` : null);
  add("target_record_id", rec.target_record_id ? `<a href="#record=${esc(rec.target_record_id)}" class="dgl-mono">${esc(rec.target_record_id)}</a>` : null);
  add("reason", esc(rec.reason));
  add("outcome", esc(rec.outcome));
  add("note", esc(rec.note));
  add("resolved_by", esc(rec.resolved_by));
  add("where", found.block !== null ? `block #${found.block}, position ${found.position}` : "pending pool (not sealed yet)");

  const about = laterRecordsAbout(id);
  const statusHtml = rec.type !== "registration" ? "" : `<h3 class="registry__panel-title" style="margin-top:18px">What happened to it later</h3>
    ${about.length ? `<table class="dgl-table">${about.map((x) => `<tr class="is-click" data-record="${esc(x.record.record_id)}"><td><span class="dgl-type dgl-type--${esc(x.record.type)}">${esc(x.record.type)}</span></td>
      <td>${esc(x.record.reason || x.record.outcome || "")}</td><td>${esc(x.record.creator_handle || x.record.resolved_by || "")}</td><td>${x.block !== null ? "block #" + x.block : "pending"}</td></tr>`).join("")}</table>`
      : '<p class="dgl-small"><span class="dgl-pill dgl-pill--ok">No revocations or disputes</span></p>'}`;

  let proof = "";
  try {
    const bundle = (await (await fetch(`/api/ledger/record/${encodeURIComponent(id)}`)).json()).proof;
    proof = DGUI.proofHtml(await DGUI.proofSteps(bundle, pinned()));
  } catch (e) { proof = '<p class="dgl-error">Could not build the proof.</p>'; }

  panel.innerHTML = `<div class="dgl-card">
    <h2 class="dgl-section__title">Record ${esc(rec.title ? "· " + rec.title : "· " + rec.type)}</h2>
    <div class="dgl-grid2">
      <div><dl class="dgl-fields">${rows.map(([k, v]) => `<dt class="dgl-mono">${k}</dt><dd>${v}</dd>`).join("")}</dl>${statusHtml}</div>
      <div><h3 class="registry__panel-title">Chain of trust, checked in your browser</h3>${proof}</div>
    </div>
    <div class="dgl-btnrow">
      ${found.block !== null ? `<a class="dgl-btn dgl-btn--indigo" href="/api/ledger/receipt/${esc(id)}">Download receipt</a>` : ""}
      <a class="dgl-btn dgl-btn--ghost" href="/ledger">Verify a file on the Ledger page</a>
      ${found.block !== null ? `<a class="dgl-btn dgl-btn--ghost" href="#block=${found.block}">Show its block</a>` : ""}
    </div>
    <details class="dgl-details"><summary>Raw record</summary><pre class="dgl-json">${esc(JSON.stringify(rec, null, 2))}</pre></details>
  </div>`;
  panel.hidden = false;
  panel.querySelectorAll("[data-record]").forEach((tr) => tr.addEventListener("click", () => { location.hash = `record=${tr.dataset.record}`; }));
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- search --

$("search").addEventListener("input", () => {
  const q = $("search").value.trim().toLowerCase();
  document.querySelectorAll("#chain .dgl-block").forEach((el) => el.classList.remove("is-hit"));
  if (q.length < 2) { $("searchResults").innerHTML = ""; return; }
  const hits = allRecords().filter((x) => ["title", "creator_handle", "record_id", "sha256_merkle_root", "type", "reason", "file_name"]
    .some((k) => String(x.record[k] || "").toLowerCase().includes(q)) || String((x.record.watermark || {}).id || "").includes(q));
  hits.forEach((x) => { if (x.block !== null) { const el = document.querySelector(`#chain .dgl-block[data-index="${x.block}"]`); if (el) el.classList.add("is-hit"); } });
  $("searchResults").innerHTML = hits.length
    ? `<table class="dgl-table">${hits.slice(0, 20).map((x) => `<tr class="is-click" data-record="${esc(x.record.record_id)}"><td><span class="dgl-type dgl-type--${esc(x.record.type)}">${esc(x.record.type)}</span></td>
        <td>${esc(x.record.title || x.record.reason || "")}</td><td>${esc(x.record.creator_handle || "")}</td><td>${x.block !== null ? "block #" + x.block : "pending"}</td></tr>`).join("")}</table>`
    : '<p class="dgl-small dgl-muted">No record matches.</p>';
  $("searchResults").querySelectorAll("[data-record]").forEach((tr) => tr.addEventListener("click", () => { location.hash = `record=${tr.dataset.record}`; }));
});

// ------------------------------------------------------------- witnesses --

async function loadWitnesses() {
  const list = S.chain.witnesses || [];
  const bl = blocks();
  S.witnessView = await Promise.all(list.map(async (w) => {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 2500);
      const data = await (await fetch(`${w.url}/witness/hashes`, { signal: ctrl.signal })).json();
      clearTimeout(t);
      let mismatch = null;
      data.hashes.forEach((h, i) => { if (mismatch === null && (!bl[i] || bl[i].block_hash !== h)) mismatch = i; });
      let lastSig = null;
      const last = data.hashes.length - 1;
      if (last >= 0) lastSig = await DG.verifySig(data.witness_public_key, data.cosignatures[last], DG.witnessStatement(last, data.hashes[last]));
      return { ...w, online: true, height: data.hashes.length, keyId: data.witness_key_id, mismatch, agrees: mismatch === null, missing: Math.max(bl.length - data.hashes.length, 0), alarms: data.alarms || [], lastSig };
    } catch (e) {
      return { ...w, online: false };
    }
  }));
  renderStats();
  $("witnesses").innerHTML = S.witnessView.map((w) => {
    if (!w.online) {
      return `<div class="dgl-card dgl-witness"><div class="dgl-witness__name">${esc(w.name)} <span class="dgl-pill dgl-pill--warn">offline</span></div>
        <p class="dgl-small dgl-muted">${esc(w.url)} did not answer. Start everything with <code>run_deepguard.py</code> to bring the witnesses up.</p></div>`;
    }
    const state = w.mismatch !== null
      ? `<span class="dgl-pill dgl-pill--bad">✗ disagrees from block #${w.mismatch}</span>`
      : w.missing ? `<span class="dgl-pill dgl-pill--warn">behind by ${w.missing} block${w.missing === 1 ? "" : "s"} — press Sync</span>`
      : '<span class="dgl-pill dgl-pill--ok">✓ agrees with every block</span>';
    return `<div class="dgl-card dgl-witness">
      <div class="dgl-witness__name">${esc(w.name)} ${state}</div>
      <dl class="dgl-fields">
        <dt>Address</dt><dd class="dgl-mono">${esc(w.url)}</dd>
        <dt>Witness key</dt><dd class="dgl-mono">${esc(w.keyId)}</dd>
        <dt>Blocks it has seen</dt><dd>${w.height} of ${bl.length}</dd>
        <dt>Its latest co-signature</dt><dd>${w.lastSig === true ? '<span class="dgl-pill dgl-pill--ok">✓ valid (checked in your browser)</span>' : w.lastSig === false ? '<span class="dgl-pill dgl-pill--bad">✗ invalid</span>' : '<span class="dgl-muted">–</span>'}</dd>
      </dl>
      ${w.alarms.length ? `<p class="dgl-note dgl-note--bad"><b>Alarm log:</b><br>${w.alarms.slice(-3).map((a) => `${esc(DGUI.time(a.at))}: ${esc(a.message)}`).join("<br>")}</p>` : ""}
    </div>`;
  }).join("");
}

async function syncWitnesses() {
  setStatus("Showing new blocks to the witnesses…");
  const d = await (await fetch("/api/ledger/witnesses/sync", { method: "POST" })).json();
  setStatus(d.witnesses.map((w) => `${esc(w.name)}: ${w.agrees ? `✓ up to date (${w.height} blocks${w.new_cosignatures ? `, ${w.new_cosignatures} new co-signature${w.new_cosignatures === 1 ? "" : "s"}` : ""})` : esc(w.alarm || "not reachable")}`).join(" · "));
  await load({ recheck: false });
  renderChain();
}

async function seal() {
  setStatus("Sealing the pending pool into a new block…");
  const d = await (await fetch("/api/ledger/seal", { method: "POST" })).json();
  if (!d.block) { setStatus(esc(d.message)); return; }
  await load({ newIndex: d.block.index });
  location.hash = `block=${d.block.index}`;
}

// --------------------------------------------------------- Bitcoin anchors --

function anchorStatusHtml(a) {
  if (a.status === "confirmed" && a.bitcoin) {
    return `<span class="dgl-pill dgl-pill--btc">✓ In Bitcoin block #${esc(a.bitcoin.height)}</span>
      <span class="dgl-small">mined ${esc(DGUI.time(a.bitcoin.time_utc))} · checked on ${esc((a.bitcoin.sources || []).join(" and "))}${a.bitcoin.single_source ? " (one explorer only)" : ""}</span>`;
  }
  if (a.status === "claimed") return '<span class="dgl-pill dgl-pill--warn">In Bitcoin, not looked up yet</span> <span class="dgl-small">press “Check for Bitcoin confirmation” while online</span>';
  if (a.status === "failed") return '<span class="dgl-pill dgl-pill--bad">✗ The proof does not reach Bitcoin</span>';
  const n = (a.calendars || []).filter((c) => c.ok).length;
  return `<span class="dgl-pill dgl-pill--warn">Waiting for Bitcoin</span> <span class="dgl-small">sent ${esc(DGUI.time(a.anchored_at))} to ${n} calendar${n === 1 ? "" : "s"}; usually confirmed within a few hours</span>`;
}

function renderAnchors() {
  const box = $("anchors");
  if (!S.anchors) { box.innerHTML = '<p class="dgl-error">Couldn’t load the anchors.</p>'; return; }
  const list = S.anchors.anchors;
  const newest = S.anchors.newest_block;
  $("anchorBtn").disabled = newest === null || S.anchors.newest_anchored;
  $("anchorBtn").textContent = S.anchors.newest_anchored ? `Block #${newest} is already anchored` : `Anchor block #${newest} to Bitcoin`;
  $("upgradeBtn").disabled = !list.length;
  if (!list.length) {
    box.innerHTML = '<p class="dgl-small dgl-muted" style="margin:14px 0 0">No block is anchored yet. Anchoring needs internet; the proof can be checked offline afterwards.</p>';
    return;
  }
  box.innerHTML = `<table class="dgl-table" style="margin-top:14px">
    <tr><th>Block</th><th>Status</th><th>Still on the chain?</th><th>Proof files</th><th></th></tr>
    ${list.slice().reverse().map((a) => `<tr>
      <td><a href="#block=${a.block_index}">#${a.block_index}</a><div class="dgl-mono dgl-muted">${esc(DG.short(a.block_hash))}</div></td>
      <td>${anchorStatusHtml(a)}</td>
      <td>${a.chain_agrees ? '<span class="dgl-pill dgl-pill--ok">✓ same block hash</span>' : '<span class="dgl-pill dgl-pill--bad">✗ the chain changed after anchoring</span>'}</td>
      <td class="dgl-small" style="white-space:nowrap"><a href="/api/ledger/anchors/${a.block_index}/header">${esc(a.header_file)}</a><br><a href="/api/ledger/anchors/${a.block_index}/proof">${esc(a.proof_file)}</a></td>
      <td>${a.status === "confirmed" ? `<button type="button" class="dgl-btn dgl-btn--ghost dgl-btn--small" data-anchor-check="${a.block_index}">Re-check in my browser</button>` : ""}</td>
    </tr><tr class="dgl-anchor-out" hidden data-anchor-out="${a.block_index}"><td colspan="5"></td></tr>`).join("")}
  </table>
  <p class="dgl-small dgl-muted">Check it without Deep-Guard: download both files for a block and drop them on opentimestamps.org,
    or run <code>python tools/verify_ledger.py anchors</code>.</p>`;
  box.querySelectorAll("[data-anchor-check]").forEach((btn) => btn.addEventListener("click", () => browserCheckAnchor(Number(btn.dataset.anchorCheck))));
}

// The anchor checked by this page alone: the block's header is re-hashed
// here, the proof's steps are replayed here, and the Bitcoin block's Merkle
// root is fetched by this browser straight from two block explorers.
async function browserCheckAnchor(index) {
  const a = anchorOf(index);
  const row = $("anchors").querySelector(`[data-anchor-out="${index}"]`);
  const cell = row.querySelector("td");
  row.hidden = false;
  cell.innerHTML = '<p class="dgl-small">Re-checking in your browser…</p>';
  const checks = [];
  const block = blocks()[index];
  const hash = block ? await DG.blockHash(block) : null;
  checks.push({ name: `Block #${index}'s header hashes to the anchored hash`, ok: hash === a.block_hash, detail: `recomputed here: ${DG.short(hash)}` });
  const path = (a.paths || []).find((p) => p.attestation.type === "bitcoin");
  if (!path) checks.push({ name: "The proof reaches Bitcoin", ok: false, detail: "no Bitcoin attestation in the proof" });
  else {
    const end = await DG.replayOts(a.block_hash, path.ops);
    if (end === null) checks.push({ name: `${path.ops.length} proof steps replayed`, ok: null, detail: "this proof uses an operation browsers can't compute; the server checked it" });
    else checks.push({ name: `${path.ops.length} proof steps replayed`, ok: end === path.msg, detail: `they end on ${DG.short(end)}` });
    try {
      const btc = await DG.bitcoinBlock(path.attestation.height);
      const ok = end !== null && end === DG.reverseHex(btc.merkle_root);
      checks.push({ name: `That is the Merkle root of Bitcoin block #${btc.height}`, ok: end === null ? null : ok,
        detail: `${btc.sources.join(" and ")} say the root is ${DG.short(btc.merkle_root)} (shown byte-reversed); mined ${DGUI.time(new Date(btc.time * 1000).toISOString())}` });
    } catch (e) {
      checks.push({ name: "Bitcoin block looked up from this browser", ok: null, detail: String(e.message || e) });
    }
  }
  cell.innerHTML = DGUI.checksHtml(checks) + `<p class="dgl-small dgl-muted">Nothing above came from the Deep-Guard server except the proof file, and a wrong proof can't end on a real Bitcoin Merkle root.</p>`;
}

$("anchorBtn").addEventListener("click", async () => {
  const btn = $("anchorBtn");
  btn.disabled = true;
  $("anchorStatus").textContent = "Sending the block hash to the OpenTimestamps calendars…";
  try {
    const r = await fetch("/api/ledger/anchor", { method: "POST" });
    if (!r.ok) { $("anchorStatus").innerHTML = `<span class="dgl-error">${esc(await DGUI.errorText(r))}</span>`; return; }
    const d = await r.json();
    $("anchorStatus").textContent = d.message;
    await load({ recheck: false });
  } finally { btn.disabled = S.anchors ? S.anchors.newest_anchored : false; }
});

$("upgradeBtn").addEventListener("click", async () => {
  const btn = $("upgradeBtn");
  btn.disabled = true;
  $("anchorStatus").textContent = "Asking the calendars and looking the Bitcoin blocks up…";
  try {
    const r = await fetch("/api/ledger/anchors/upgrade", { method: "POST" });
    if (!r.ok) { $("anchorStatus").innerHTML = `<span class="dgl-error">${esc(await DGUI.errorText(r))}</span>`; return; }
    const d = await r.json();
    $("anchorStatus").textContent = d.results.length
      ? d.results.map((x) => `Block #${x.block_index}: ${x.status}. ${(x.notes || []).slice(-1)[0] || ""}`).join(" · ")
      : "Every anchor is already confirmed.";
    await load({ recheck: false });
  } finally { btn.disabled = false; }
});

// ------------------------------------------------------------ tamper lab --

let ATTACKS = {};
async function loadAttacks() {
  ATTACKS = (await (await fetch("/api/ledger/attacks")).json()).attacks;
  const names = { edit_record: "1 · Edit an old record", edit_and_rehash: "2 · Edit it and recompute the hashes (no key)", delete_block: "3 · Delete a block", insider_rewrite: "4 · Insider rewrite with the real key", collusion_rewrite: "5 · Insider rewrite, and the witnesses collude" };
  $("attack").innerHTML = Object.keys(ATTACKS).map((k) => `<option value="${k}">${esc(names[k] || k)}</option>`).join("");
  $("attackHint").textContent = ATTACKS[$("attack").value];
}
$("attack").addEventListener("change", () => { $("attackHint").textContent = ATTACKS[$("attack").value]; });

function fillTamperBlocks() {
  const bl = blocks();
  const current = $("attackBlock").value;
  $("attackBlock").innerHTML = bl.map((b, i) => `<option value="${i}">#${i} ${esc(blockLabel(b))}</option>`).join("");
  if (current && Number(current) < bl.length) $("attackBlock").value = current;
  else if (bl.length > 1) $("attackBlock").value = "1";
}

$("tamperForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("tamperBtn");
  btn.disabled = true;
  try {
    const r = await fetch("/api/ledger/tamper-lab", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attack: $("attack").value, block_index: Number($("attackBlock").value) }) });
    if (!r.ok) { $("tamperResult").innerHTML = `<p class="dgl-error">${esc(await DGUI.errorText(r))}</p>`; return; }
    const d = await r.json();
    // the browser checks the tampered copy too — it should agree with the server
    const mine = await DG.checkChain(d.blocks, pinned());
    const bad = d.report.first_bad_index;
    const caught = !d.report.ok || d.witnesses.some((w) => w.first_mismatch !== null && w.first_mismatch !== undefined)
      || (d.anchors || []).some((x) => !x.agrees);
    $("tamperResult").innerHTML = `
      <p class="dgl-note ${caught ? "dgl-note--good" : ""}" style="margin-top:16px"><b>What the attacker did:</b> ${esc(d.description)}<br><b>Result:</b> ${esc(d.verdict)}</p>
      <div class="dgl-chainwrap"><div class="dgl-chain">${chainHtml(d.blocks, d.report, { mini: true, forged: new Set(d.blocks.map((b, i) => (blocks()[i] && blocks()[i].block_hash === b.block_hash ? -1 : i)).filter((i) => i >= 0)) })}</div></div>
      <p class="dgl-small dgl-muted">${d.attack === "collusion_rewrite"
        ? "Amber blocks have a different hash from the real chain. Colluding witnesses would sign them anyway; an anchor made before the rewrite still holds the original hash."
        : "Amber blocks have a different hash from the real chain — the one the witnesses wrote down."}</p>
      <div class="dgl-grid3">
        <div><h3 class="registry__panel-title">${bad !== null ? `Block #${bad} on the tampered copy` : "The tampered copy"}</h3>
          ${bad !== null ? DGUI.checksHtml(d.report.blocks[bad].checks) : '<p class="dgl-small">Every block passes on its own.</p>'}
          <p class="dgl-small dgl-muted">Your browser checked the same copy and ${mine.ok === d.report.ok && mine.first_bad_index === bad ? "agrees" : "DISAGREES"} with the server.</p></div>
        <div><h3 class="registry__panel-title">The witnesses</h3>
          ${DGUI.checksHtml(d.witnesses.map((w) => w.colluding ? { name: `${w.name} goes along with it`, ok: null, detail: "colluding (simulated): it signs whatever it is shown" }
            : !w.online ? { name: `${w.name} is offline`, ok: null, detail: "start it to see it catch rewrites" }
            : w.first_mismatch !== null && w.first_mismatch !== undefined ? { name: `${w.name} catches it`, ok: true, detail: `its notebook holds a different hash for block #${w.first_mismatch}` }
            : { name: `${w.name} sees no difference`, ok: null, detail: w.witness_height ? `it holds ${w.witness_height} blocks and they all still match` : "it has not seen any blocks yet — press Sync witnesses" }))}
        </div>
        <div><h3 class="registry__panel-title">The Bitcoin anchors</h3>
          ${(d.anchors || []).length ? DGUI.checksHtml(d.anchors.map((x) => x.agrees
            ? { name: `Anchor of block #${x.block_index} sees no difference`, ok: null, detail: x.block_index < Number($("attackBlock").value) ? "the rewrite is newer than this anchor" : "the anchored hash still matches" }
            : { name: `Anchor of block #${x.block_index} catches it`, ok: true, detail: x.status === "confirmed" ? `Bitcoin block #${x.bitcoin_height} holds the original hash` : "the calendars hold the original hash (Bitcoin confirmation pending)" }))
            : '<p class="dgl-small dgl-muted">No block is anchored yet. Anchor one above to protect against attack 5.</p>'}
        </div>
      </div>
      <p class="dgl-small dgl-muted">Your real ledger was not changed. Reload the page: it is still intact.</p>`;
  } finally {
    btn.disabled = false;
  }
});

// --------------------------------------------------------------- receipts --

$("receiptFile").addEventListener("change", async () => {
  const file = $("receiptFile").files[0];
  if (!file) return;
  let receipt;
  try { receipt = JSON.parse(await file.text()); } catch (e) { $("receiptResult").innerHTML = '<p class="dgl-error">That is not a JSON file.</p>'; return; }
  if (!receipt.record || !receipt.block_header) { $("receiptResult").innerHTML = '<p class="dgl-error">That JSON is not a Deep-Guard receipt.</p>'; return; }
  const result = await DG.verifyReceipt(receipt, pinned());
  const onChain = blocks()[receipt.block_header.index];
  const stillThere = onChain && onChain.block_hash === receipt.block_hash;
  $("receiptResult").innerHTML = `
    <p style="margin-top:14px"><span class="dgl-pill ${result.ok ? "dgl-pill--ok" : "dgl-pill--bad"}">${result.ok ? "✓ receipt is genuine" : "✗ receipt does not check out"}</span>
      &nbsp;“${esc(receipt.record.title || receipt.record.type)}” in block #${esc(receipt.block_header.index)}</p>
    ${DGUI.checksHtml([...result.checks, { name: stillThere ? `Block #${receipt.block_header.index} on this ledger still has the same hash` : `This ledger's block #${receipt.block_header.index} has a DIFFERENT hash now`, ok: stillThere, detail: stillThere ? "" : "the ledger was rewritten after this receipt was issued, or the receipt is from another ledger" }])}`;
});

// ------------------------------------------------------------------ keys --

function renderKeys() {
  const counts = {};
  allRecords().forEach((x) => { if (x.record.creator_id) counts[x.record.creator_id] = (counts[x.record.creator_id] || 0) + 1; });
  $("keys").innerHTML = `
    <dl class="dgl-fields"><dt>Deep-Guard's key ID</dt><dd class="dgl-mono">${esc(S.chain.service_key_id)}</dd></dl>
    <details class="dgl-details"><summary>Deep-Guard's public key (what every block signature is checked against)</summary><pre class="dgl-json">${esc(S.chain.service_public_key)}</pre></details>
    ${S.creators.length ? `<table class="dgl-table" style="margin-top:14px"><tr><th>Creator</th><th>Key ID</th><th>Records filed</th><th>Key created</th></tr>
      ${S.creators.map((c) => `<tr><td>${esc(c.handle)}</td><td class="dgl-mono">${esc(c.key_id)}</td><td>${counts[c.creator_id] || 0}</td><td>${esc(DGUI.time(c.created_at))}</td></tr>`).join("")}</table>`
      : '<p class="dgl-small dgl-muted" style="margin-top:12px">No creators yet — register a file on the Ledger page.</p>'}`;
}

// --------------------------------------------------------------- routing --

async function route() {
  const h = location.hash.slice(1);
  const [k, v] = h.split("=");
  if (k === "block" && v !== undefined) await selectBlock(Number(v));
  if (k === "record" && v) await selectRecord(decodeURIComponent(v));
}
window.addEventListener("hashchange", route);

$("browserCheckBtn").addEventListener("click", async () => { await load({ recheck: false }); await browserCheck(); });
$("serverCheckBtn").addEventListener("click", serverCheck);
$("syncBtn").addEventListener("click", syncWitnesses);
$("sealBtn").addEventListener("click", seal);
$("reloadBtn").addEventListener("click", () => load());

(async () => {
  try {
    await load();
    await loadAttacks();
    await route();
  } catch (e) {
    setStatus(`<span class="dgl-error">Could not load the ledger: ${esc(e.message || e)}. Is the server running?</span>`);
  }
})();
