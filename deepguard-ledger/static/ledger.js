// Deep-Guard — /ledger page: register, verify (four verdicts), corrections.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = DGUI.esc;
let SUMMARY = null;

async function loadSummary() {
  const r = await fetch("/api/ledger/summary");
  SUMMARY = await r.json();
}

async function loadCreators() {
  try {
    const d = await (await fetch("/api/ledger/creators")).json();
    $("creatorList").innerHTML = d.creators.map((c) => `<option value="${esc(c.handle)}">`).join("");
  } catch (e) { /* the datalist is a convenience only */ }
}

function showError(el, message) {
  el.textContent = message;
  el.hidden = !message;
}

function pct(p) { return p === null || p === undefined ? "–" : `${(p * 100).toFixed(1)}%`; }

function copyButton(text) {
  return `<button type="button" class="dgl-btn dgl-btn--ghost dgl-btn--small" data-copy="${esc(text)}">Copy</button>`;
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copy]");
  if (!btn) return;
  navigator.clipboard.writeText(btn.dataset.copy).then(() => { btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy"), 1200); });
});

function fingerprintRows(rec) {
  const rows = [
    ["SHA-256 (exact file)", `<span class="dgl-mono">${esc(rec.sha256_merkle_root)}</span>`],
  ];
  if (rec.phash) rows.push(["pHash (looks-like, 256 bits)", `<span class="dgl-mono">${esc(rec.phash)}</span>`]);
  if (rec.phash_frames) rows.push(["pHash per frame", `${rec.phash_frames.length} frames sampled across the video`]);
  if (rec.content_vector_sha256) rows.push(["SSCD content vector", `512 numbers kept beside the ledger; their hash is on it: <span class="dgl-mono">${esc(DG.short(rec.content_vector_sha256))}</span>`]);
  if (rec.video) rows.push(["Video", `${esc(rec.video.resolution)} · ${esc(rec.video.duration)}`]);
  rows.push(["Size", `${(rec.file_size_bytes / 1024).toFixed(1)} KB`]);
  return rows;
}

function fieldsHtml(rows) {
  return '<dl class="dgl-fields">' + rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("") + "</dl>";
}

function witnessHtml(witnesses) {
  if (!witnesses || !witnesses.length) return "";
  return '<div class="dgl-btnrow">' + witnesses.map((w) => {
    if (w.agrees && w.online) return `<span class="dgl-pill dgl-pill--ok"><span class="dgl-dot"></span>${esc(w.name)} co-signed</span>`;
    if (!w.online) return `<span class="dgl-pill dgl-pill--warn" title="${esc(w.alarm || "")}"><span class="dgl-dot"></span>${esc(w.name)} offline</span>`;
    return `<span class="dgl-pill dgl-pill--bad" title="${esc(w.alarm || "")}"><span class="dgl-dot"></span>${esc(w.name)}: ${esc(w.alarm || "disagrees")}</span>`;
  }).join("") + "</div>";
}

// ------------------------------------------------------------- register --

$("registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError($("regError"), "");
  const file = $("regFile").files[0];
  if (!file) return showError($("regError"), "Choose a file first.");
  const body = new FormData();
  body.append("file", file);
  body.append("title", $("regTitle").value);
  body.append("creator", $("regCreator").value);
  body.append("declared_origin", document.querySelector('input[name="origin"]:checked').value);
  body.append("seal_now", $("regSeal").checked ? "true" : "false");
  const btn = $("regBtn");
  btn.disabled = true;
  btn.textContent = file.type.startsWith("video/") ? "Fingerprinting video frames…" : "Fingerprinting & sealing…";
  try {
    const r = await fetch("/api/ledger/register", { method: "POST", body });
    if (!r.ok) return showError($("regError"), await DGUI.errorText(r));
    await renderRegistered(await r.json());
    loadCreators();
  } catch (err) {
    showError($("regError"), "Couldn't reach the server. Is it still running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Register on the ledger";
  }
});

async function renderRegistered(d) {
  if (!SUMMARY) await loadSummary();
  const rec = d.record;
  const sealed = !!d.block;
  const rows = [
    ["Record ID", `<span class="dgl-mono">${esc(rec.record_id)}</span> ${copyButton(rec.record_id)}`],
    ["Title", esc(rec.title)],
    ["Creator", `${esc(d.creator.handle)} <span class="dgl-muted">· key ${esc(d.creator.key_id)}</span>${d.creator_created ? ' <span class="dgl-pill dgl-pill--mod">new key pair created</span>' : ""}`],
    ["Declared origin", esc(DGUI.originLabel(rec.declared_origin))],
    ["Media", esc(rec.media_type) + " · " + esc(rec.file_name)],
    ...fingerprintRows(rec),
  ];
  if (d.detector) {
    rows.push(["AI detector at the door", `${pct(d.detector.probability_fake)} AI${d.detector.calibrated_probability_fake !== null ? ` <span class="dgl-muted">(calibrated ${pct(d.detector.calibrated_probability_fake)})</span>` : ""} <span class="dgl-muted">— written down beside the claim, not a judgement</span>`]);
  }
  if (sealed) rows.push(["Block", `#${d.block.index} · hash <span class="dgl-mono">${esc(DG.short(d.block.block_hash))}</span> · ${esc(DGUI.time(d.block.timestamp))}`]);

  let proof = "";
  if (d.proof) {
    const steps = await DGUI.proofSteps(d.proof, SUMMARY.service_public_key);
    proof = `<h3 class="registry__panel-title" style="margin-top:22px">Proof, checked in your browser</h3>
      <p class="dgl-small dgl-muted">Each link below was recomputed by this page with the browser's own crypto, not taken from the server.</p>
      ${DGUI.proofHtml(steps)}`;
  }

  const section = $("resultSection");
  section.innerHTML = `
    <div class="dgl-verdict dgl-verdict--1">
      <div class="dgl-verdict__kicker">Registered</div>
      <div class="dgl-verdict__headline">${sealed ? `Written to block #${d.block.index}` : "Waiting in the pending pool"}</div>
      <p class="dgl-verdict__text">${sealed
        ? "The record is signed by you, sealed into a block signed by Deep-Guard, and chained to every block before it."
        : "The record is signed by you and waiting. Seal the pending pool from the explorer to put it in a block."}</p>
      ${d.warning ? `<p class="dgl-note">${esc(d.warning)}</p>` : ""}
      ${fieldsHtml(rows)}
      ${witnessHtml(d.witnesses)}
      ${proof}
      <div class="dgl-btnrow">
        <a class="dgl-btn dgl-btn--indigo" href="/explorer#record=${esc(rec.record_id)}">See it in the explorer</a>
        ${sealed ? `<a class="dgl-btn dgl-btn--ghost" href="/api/ledger/receipt/${esc(rec.record_id)}">Download receipt</a>` : ""}
      </div>
      <details class="dgl-details"><summary>The full signed record</summary><pre class="dgl-json">${esc(JSON.stringify(rec, null, 2))}</pre></details>
    </div>`;
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --------------------------------------------------------------- verify --

$("verifyForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError($("verError"), "");
  const file = $("verFile").files[0];
  if (!file) return showError($("verError"), "Choose a file first.");
  const body = new FormData();
  body.append("file", file);
  body.append("ai_fallback", $("verAi").checked ? "true" : "false");
  const btn = $("verBtn");
  btn.disabled = true;
  btn.textContent = "Searching the ledger…";
  try {
    const r = await fetch("/api/ledger/verify", { method: "POST", body });
    if (!r.ok) return showError($("verError"), await DGUI.errorText(r));
    await renderVerdict(await r.json());
  } catch (err) {
    showError($("verError"), "Couldn't reach the server. Is it still running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Check against the ledger";
  }
});

function howMatched(m, thresholds) {
  if (m.kind === "exact") return "Exact: SHA-256 identical — the very same bytes";
  if (m.kind === "visual") return `Looks-like: pHash differs by ${m.phash_bits_different} of 256 bits (match ≤ ${thresholds.phash_bits})`;
  return `Content: SSCD similarity ${m.content_similarity} (match ≥ ${thresholds.content_similarity}) — a crop or edit of it`;
}

function statusHtml(status) {
  const parts = [];
  if (status.revoked) parts.push(`<span class="dgl-pill dgl-pill--bad">Revoked by its creator ${esc(DGUI.time(status.revoked.filed_at))}</span>`);
  for (const d of status.disputes) parts.push(`<span class="dgl-pill dgl-pill--warn" title="${esc(d.reason)}">Disputed by ${esc(d.creator_handle)}</span>`);
  for (const r of status.resolutions) parts.push(`<span class="dgl-pill dgl-pill--mod">Dispute ${esc(r.outcome)}</span>`);
  return parts.length ? parts.join(" ") : '<span class="dgl-pill dgl-pill--ok">No revocations or disputes</span>';
}

async function renderVerdict(d) {
  if (!SUMMARY) await loadSummary();
  document.querySelectorAll(".dgl-four__item").forEach((el) => el.classList.toggle("is-on", el.classList.contains("v" + d.verdict_number)));
  const parts = [];
  parts.push(`
    <div class="dgl-verdict dgl-verdict--${d.verdict_number}">
      <div class="dgl-verdict__kicker">Verdict ${d.verdict_number} of 4 · ${esc(d.upload.file_name)}</div>
      <div class="dgl-verdict__headline">${esc(d.headline)}</div>
      <p class="dgl-verdict__text">${esc(d.explanation)}</p>
      ${d.notes.map((n) => `<p class="dgl-note ${n.startsWith("WARNING") ? "dgl-note--bad" : ""}">${esc(n)}</p>`).join("")}
    </div>`);

  const m = d.match;
  if (m) {
    const rec = m.record;
    const rows = [
      ["Title", esc(rec.title)],
      ["Registered by", `${esc(rec.creator_handle)} <span class="dgl-muted">· key ${esc(rec.creator_key_id)}</span>`],
      ["Declared origin", esc(DGUI.originLabel(rec.declared_origin))],
      ["Registered at", esc(DGUI.time(rec.registered_at))],
      ["Where", m.sealed ? `Block #${m.block_index}, position ${m.position}` : "Pending pool (not sealed yet)"],
      ["How it matched", esc(howMatched(m, d.thresholds))],
      ["Status", statusHtml(m.status)],
      ["Record ID", `<span class="dgl-mono">${esc(rec.record_id)}</span> ${copyButton(rec.record_id)}`],
      ["Your file's SHA-256", `<span class="dgl-mono">${esc(d.upload.sha256_merkle_root)}</span>`],
      ["Registered SHA-256", `<span class="dgl-mono">${esc(rec.sha256_merkle_root)}</span> ${rec.sha256_merkle_root === d.upload.sha256_merkle_root ? '<span class="dgl-pill dgl-pill--ok">identical</span>' : '<span class="dgl-pill dgl-pill--mod">different file</span>'}`],
    ];
    if (rec.detector_reading) rows.push(["AI detector when registered", `${esc(rec.detector_reading.ai_probability_percent)}% AI`]);
    let proof = "";
    if (m.sealed) {
      const steps = await DGUI.proofSteps(m.proof, SUMMARY.service_public_key);
      proof = `<div class="dgl-grid2" style="margin-top:18px">
        <div><h3 class="registry__panel-title">Proof, checked in your browser</h3>${DGUI.proofHtml(steps)}</div>
        <div><h3 class="registry__panel-title">Block #${m.block_index}, checked by the server</h3>${DGUI.checksHtml(m.block_check.checks)}</div>
      </div>`;
    }
    parts.push(`<div class="dgl-card" style="margin-top:18px">
      <h3 class="registry__panel-title">What it matched</h3>
      ${fieldsHtml(rows)}
      ${proof}
      <div class="dgl-btnrow">
        <a class="dgl-btn dgl-btn--indigo" href="/explorer#record=${esc(rec.record_id)}">Open in the explorer</a>
        ${m.sealed ? `<a class="dgl-btn dgl-btn--ghost" href="/api/ledger/receipt/${esc(rec.record_id)}">Download receipt</a>` : ""}
      </div>
    </div>`);
  }

  if (d.other_matches && d.other_matches.length) {
    parts.push(`<div class="dgl-card" style="margin-top:18px"><h3 class="registry__panel-title">Other records it also resembles</h3>
      <table class="dgl-table"><tr><th>Title</th><th>Creator</th><th>How</th><th>Status</th></tr>
      ${d.other_matches.map((o) => `<tr><td>${esc(o.record.title)}</td><td>${esc(o.record.creator_handle)}</td><td>${esc(howMatched(o, d.thresholds))}</td><td>${statusHtml(o.status)}</td></tr>`).join("")}
      </table></div>`);
  }

  if (d.detector) {
    const cal = d.detector.calibrated_probability_fake;
    parts.push(`<div class="dgl-card" style="margin-top:18px;border-style:dashed">
      <h3 class="registry__panel-title">AI detector fallback &mdash; a prediction, not proof</h3>
      <p class="dgl-small">No record exists, so the bouncer's detector was asked for its guess.</p>
      ${fieldsHtml([
        ["Raw reading", `${pct(d.detector.probability_fake)} likely AI-generated`],
        ["Calibrated reading", cal === null ? "–" : `${pct(cal)} <span class="dgl-muted">(temperature-scaled: what the raw number honestly means)</span>`],
        ["Frames scored", String(d.detector.frames_scored)],
      ])}
      <p class="dgl-note">This detector catches only about 58% of today's AI images and false-alarms on about 14% of real photos
        (see the <a href="/bias-map">Bias Map</a>). That is exactly why provenance comes first.</p>
    </div>`);
  } else if (d.verdict_number === 3) {
    parts.push(`<p class="dgl-note">No record found, and the AI fallback was switched off for this check.</p>`);
  }

  const section = $("resultSection");
  section.innerHTML = parts.join("");
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------- corrections --

function wireCorrection(formId, url) {
  const form = $(formId);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const out = form.querySelector("[data-out]");
    const btn = form.querySelector("button");
    out.className = "dgl-small";
    out.textContent = "Writing to the ledger…";
    btn.disabled = true;
    try {
      const r = await fetch(url, { method: "POST", body: new FormData(form) });
      if (!r.ok) {
        out.className = "dgl-small dgl-error";
        out.textContent = await DGUI.errorText(r);
        return;
      }
      const d = await r.json();
      out.innerHTML = `Written to <a href="/explorer#block=${d.block.index}">block #${d.block.index}</a> as record
        <span class="dgl-mono">${esc(d.record.record_id.slice(0, 12))}…</span> ${copyButton(d.record.record_id)}`;
      form.reset();
      loadCreators();
    } catch (err) {
      out.className = "dgl-small dgl-error";
      out.textContent = "Couldn't reach the server.";
    } finally {
      btn.disabled = false;
    }
  });
}

wireCorrection("revokeForm", "/api/ledger/revoke");
wireCorrection("disputeForm", "/api/ledger/dispute");
wireCorrection("resolveForm", "/api/ledger/resolve");

loadSummary();
loadCreators();
