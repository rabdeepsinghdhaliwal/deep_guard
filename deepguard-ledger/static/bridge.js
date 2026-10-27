// Deep-Guard ledger bridge — added to the bouncer's original pages at serve
// time by ledger_app.py (the page files themselves are never edited).
//   * every page: "Ledger" and "Explorer" in the menu
//   * analysis page: after a result appears, a provenance card that checks
//     the same file against the ledger and offers to register it
"use strict";

(function () {
  const esc = (s) => String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function addNav() {
    const header = document.querySelector(".site-header");
    if (!header || header.querySelector('[href="/ledger"]')) return;
    let nav = header.querySelector(".site-nav");
    if (!nav) {
      // registry / bias-map pages have a single "back" link: gather it into a nav
      nav = document.createElement("nav");
      nav.className = "site-nav";
      const back = header.querySelector(".site-nav__link");
      header.appendChild(nav);
      if (back) nav.appendChild(back);
    }
    const before = nav.querySelector(".status");
    for (const [href, label] of [["/ledger", "Ledger"], ["/explorer", "Explorer"]]) {
      const a = document.createElement("a");
      a.className = "site-nav__link";
      a.href = href;
      a.textContent = label;
      nav.insertBefore(a, before || null);
    }
  }

  function currentFile() {
    try {
      // script.js keeps the chosen file in a top-level `let selectedFile`
      // eslint-disable-next-line no-undef
      if (typeof selectedFile !== "undefined" && selectedFile) return selectedFile;
    } catch (e) { /* not on the analysis page */ }
    const input = document.getElementById("fileInput");
    return input && input.files ? input.files[0] : null;
  }

  const VERDICT_COLOURS = { 1: "var(--true)", 2: "#2F6BA8", 3: "var(--unsure)", 4: "#6A4AB0" };

  async function errorText(r) {
    try { const b = await r.json(); return typeof b.detail === "string" ? b.detail : (b.detail && b.detail.message) || JSON.stringify(b.detail || b); }
    catch (e) { return `Request failed (HTTP ${r.status}).`; }
  }

  function provenanceCard() {
    const result = document.getElementById("result");
    if (!result) return;
    let card = document.getElementById("dglProvenance");
    let lastFile = null;

    const observer = new MutationObserver(() => {
      const file = currentFile();
      if (result.hidden) {
        if (card) card.hidden = true;
        return;
      }
      if (!file || file === lastFile) { if (card) card.hidden = false; return; }
      lastFile = file;
      if (!card) {
        card = document.createElement("section");
        card.id = "dglProvenance";
        card.className = "dgl-bridge";
        result.insertAdjacentElement("afterend", card);
      }
      card.hidden = false;
      check(card, file);
    });
    observer.observe(result, { attributes: true, attributeFilter: ["hidden", "data-verdict"] });
  }

  async function check(card, file) {
    card.innerHTML = `<div class="dgl-bridge__kicker">Provenance · Deep-Guard ledger</div>
      <div class="dgl-bridge__head">Checking the ledger for this file…</div>
      <p>The verdict above is the detector's guess. The ledger can do better: if this file was registered, it can prove it.</p>`;
    const body = new FormData();
    body.append("file", file);
    body.append("ai_fallback", "false");
    let d;
    try {
      const r = await fetch("/api/ledger/verify", { method: "POST", body });
      if (!r.ok) { card.innerHTML += `<p class="dgl-error">${esc(await errorText(r))}</p>`; return; }
      d = await r.json();
    } catch (e) { card.innerHTML += '<p class="dgl-error">Couldn\'t reach the ledger.</p>'; return; }

    const m = d.match;
    const matchLine = m
      ? `Matches <b>“${esc(m.record.title)}”</b> registered by <b>${esc(m.record.creator_handle)}</b>
         ${m.sealed ? `in block #${m.block_index}` : "(pending)"} · ${m.kind === "exact" ? "exact same file" : m.kind === "visual" ? `looks the same (${m.phash_bits_different}/256 bits differ)` : `same content (similarity ${m.content_similarity})`}.`
      : "Nothing on the ledger matches this file, so the detector's guess above is all there is.";
    card.innerHTML = `
      <div class="dgl-bridge__kicker">Provenance · Deep-Guard ledger · verdict ${d.verdict_number} of 4</div>
      <div class="dgl-bridge__head" style="color:${VERDICT_COLOURS[d.verdict_number]}">${esc(d.headline)}</div>
      <p>${matchLine}</p>
      ${d.notes.map((n) => `<p class="dgl-note">${esc(n)}</p>`).join("")}
      <div class="dgl-btnrow">
        ${m ? `<a class="dgl-btn dgl-btn--indigo dgl-btn--small" href="/explorer#record=${esc(m.record.record_id)}">See the proof in the explorer</a>` : ""}
        <a class="dgl-btn dgl-btn--ghost dgl-btn--small" href="/ledger">Open the Ledger page</a>
      </div>
      ${d.verdict_number === 1 ? "" : `
      <form id="dglRegister">
        <label class="dgl-span2" style="font-weight:600;font-size:13px">Register this file on the ledger</label>
        <input type="text" name="title" placeholder="Title" required maxlength="200" />
        <input type="text" name="creator" placeholder="Your creator name" required maxlength="40" />
        <select name="declared_origin">
          <option value="camera">Camera original</option>
          <option value="edited">Edited by me</option>
          <option value="ai-generated">AI-generated (declared)</option>
        </select>
        <button class="dgl-btn dgl-btn--small" type="submit">Register &amp; seal</button>
        <p class="dgl-span2 dgl-small" data-out></p>
      </form>`}`;
    const form = card.querySelector("#dglRegister");
    if (!form) return;
    form.querySelectorAll("input, select").forEach((el) => Object.assign(el.style, { font: "14px var(--sans)", padding: "8px 10px", border: "1px solid var(--rule-strong)", borderRadius: "3px", background: "var(--paper)" }));
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const out = form.querySelector("[data-out]");
      const data = new FormData(form);
      data.append("file", file);
      data.append("seal_now", "true");
      out.textContent = "Fingerprinting, signing and sealing…";
      const r = await fetch("/api/ledger/register", { method: "POST", body: data });
      if (!r.ok) { out.innerHTML = `<span class="dgl-error">${esc(await errorText(r))}</span>`; return; }
      const reg = await r.json();
      const wit = (reg.witnesses || []).filter((w) => w.agrees).map((w) => w.name).join(" and ");
      out.innerHTML = `<span class="dgl-pill dgl-pill--ok">✓ Written to block #${reg.block.index}</span> ${wit ? `co-signed by ${esc(wit)}` : ""}
        · <a href="/explorer#record=${esc(reg.record.record_id)}">see it in the explorer</a>
        · <a href="/api/ledger/receipt/${esc(reg.record.record_id)}">download receipt</a>`;
    });
  }

  addNav();
  provenanceCard();
})();
