// Deep-Guard — the Content Credentials (C2PA) card, drawn from the summary
// credentials.py returns. Shared by the Ledger page and the provenance card
// bridge.js adds under every analysis result.
"use strict";

window.DGCR = (() => {
  const esc = (s) => String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // How strongly each answer reads: the pill and the card's edge colour.
  const TONE = { trusted: "ok", ours: "ok", unknown: "warn", expired: "warn", altered: "bad", problem: "bad", error: "bad", none: "none" };
  const LABEL = {
    trusted: "Signed · on a C2PA trust list", ours: "Signed · by this Deep-Guard server", unknown: "Signed · signer not proven",
    expired: "Certificate expired", altered: "Changed after signing", problem: "Can't be checked", error: "Unreadable", none: "None",
  };
  const PILL = { ok: "dgl-pill--ok", warn: "dgl-pill--warn", bad: "dgl-pill--bad", none: "" };

  function cardHtml(c) {
    if (!c) return "";
    const tone = TONE[c.verdict] || "none";
    if (c.verdict === "none") {
      return `<div class="dgl-cr-card is-none"><span class="dgl-cr" aria-hidden="true">cr</span>
        <div><b>No Content Credentials</b><p class="dgl-small dgl-muted">${esc(c.headline)}</p></div></div>`;
    }
    const rows = [];
    if (c.signer) rows.push(["Signed by", esc(c.signer) + (c.signed_at ? ` <span class="dgl-muted">· ${esc(new Date(c.signed_at).toLocaleString())}</span>` : "")]);
    if (c.claim_generator) rows.push(["Made with", esc(c.claim_generator)]);
    if (c.source_type_text) rows.push(["Declared as", esc(c.source_type_text) + (c.ai_declared ? ' <span class="dgl-pill dgl-pill--ai">AI</span>' : "")]);
    if (c.actions && c.actions.length) rows.push(["History", c.actions.map((a) => esc(String(a.action || "").replace(/^c2pa\./, ""))).join(" → ")]);
    if (c.soft_binding) rows.push(["Watermark it names", `<span class="dgl-mono">${esc(c.soft_binding.alg)}</span>`]);
    if (c.deepguard && c.deepguard.record_id) rows.push(["Deep-Guard record", `<a class="dgl-mono" href="/explorer#record=${esc(c.deepguard.record_id)}">${esc(String(c.deepguard.record_id).slice(0, 12))}…</a>`]);
    if (c.issues && c.issues.length) rows.push(["Issues", c.issues.map((i) => esc(i.text)).join("; ")]);
    return `<div class="dgl-cr-card is-${tone}">
      <span class="dgl-cr" aria-hidden="true">cr</span>
      <div style="min-width:0;flex:1">
        <div class="dgl-cr-card__head"><b>Content Credentials</b> <span class="dgl-pill ${PILL[tone]}">${esc(LABEL[c.verdict] || c.verdict)}</span></div>
        <p class="dgl-small">${esc(c.headline)}</p>
        ${rows.length ? `<dl class="dgl-fields">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>` : ""}
      </div></div>`;
  }
  return { cardHtml };
})();
