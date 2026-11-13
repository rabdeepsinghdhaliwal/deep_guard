// Deep-Guard Lens — the badge pinned on a checked picture.
//
// Injected only when a check starts (never on every page). It draws inside a
// shadow root, so the website's styles can't change the badge and the badge
// can't change the website. It only displays what background.js sends.
"use strict";

(() => {
  if (window.__deepGuardLens) return;
  window.__deepGuardLens = true;

  const VERDICT = {
    1: { label: "Authentic & unaltered", tone: "v1" },
    2: { label: "Same picture, file modified", tone: "v2" },
    3: { label: "No provenance found", tone: "v3" },
    4: { label: "Declared AI-generated", tone: "v4" },
  };
  const CR_TEXT = {
    trusted: "signed, signer on a C2PA trust list", ours: "signed by this Deep-Guard server",
    unknown: "signed, signer not proven", expired: "signer's certificate expired", altered: "changed after signing",
    problem: "present but can't be checked", error: "unreadable", none: "no Content Credentials",
  };

  const host = document.createElement("div");
  host.setAttribute("data-deepguard-lens", "");
  host.style.cssText = "position:absolute;left:0;top:0;width:0;height:0;z-index:2147483647;";
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>
    :host { all: initial; }
    .b { position: absolute; max-width: 320px; min-width: 210px; box-sizing: border-box; font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         color: #1C1A17; background: #FFFFFF; border: 1px solid #D2CBBE; border-left: 5px solid #8A837A; border-radius: 6px;
         box-shadow: 0 6px 22px rgba(0,0,0,.28); padding: 9px 11px 9px 10px; }
    .b.v1 { border-left-color: #2E7D5B; } .b.v2 { border-left-color: #2F6BA8; } .b.v3 { border-left-color: #B07A1E; } .b.v4 { border-left-color: #6A4AB0; } .b.err { border-left-color: #B23B34; }
    .top { display: flex; align-items: center; gap: 7px; }
    .mark { width: 18px; height: 18px; flex: 0 0 auto; }
    .brand { font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #55504A; }
    .x { margin-left: auto; border: 0; background: none; font-size: 16px; line-height: 1; cursor: pointer; color: #8A837A; padding: 0 2px; }
    .x:hover { color: #1C1A17; }
    .head { font-weight: 700; font-size: 14px; margin: 5px 0 2px; }
    .v1 .head { color: #2E7D5B; } .v2 .head { color: #2F6BA8; } .v3 .head { color: #8E6114; } .v4 .head { color: #6A4AB0; } .err .head { color: #B23B34; }
    .line { margin: 3px 0 0; color: #3A3632; }
    .k { color: #8A837A; }
    .cr { display: inline-block; font-weight: 700; font-size: 10px; background: #1C1A17; color: #fff; border-radius: 4px; padding: 1px 4px; margin-right: 4px; vertical-align: 1px; }
    .mono { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
    a { color: #3A4CB4; text-decoration: none; font-weight: 600; } a:hover { text-decoration: underline; }
    .links { margin-top: 6px; display: flex; gap: 12px; }
    .spin { display: inline-block; width: 11px; height: 11px; border: 2px solid #D2CBBE; border-top-color: #3A4CB4; border-radius: 50%; animation: s .8s linear infinite; vertical-align: -1px; margin-right: 6px; }
    @keyframes s { to { transform: rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) { .spin { animation: none; } }
  </style>`;

  const esc = (s) => String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const MARK = `<svg class="mark" viewBox="0 0 26 26" fill="none" aria-hidden="true"><path d="M13 2.5 3.5 6.2v6.4c0 5.6 3.9 10.2 9.5 11.4 5.6-1.2 9.5-5.8 9.5-11.4V6.2L13 2.5Z" stroke="#3A4CB4" stroke-width="2" stroke-linejoin="round"/><path d="M9.3 12.8l2.6 2.6 5-5" stroke="#3A4CB4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const badges = new Map();  // check id -> { el, img }

  function findImage(srcUrl) {
    const all = [...document.images];
    return all.find((i) => i.currentSrc === srcUrl) || all.find((i) => i.src === srcUrl) || null;
  }

  function place(b) {
    if (!b.img || !b.img.isConnected) { b.el.style.display = "none"; return; }
    const r = b.img.getBoundingClientRect();
    b.el.style.display = r.width < 2 ? "none" : "";
    b.el.style.left = `${Math.max(r.left + window.scrollX + 8, window.scrollX + 4)}px`;
    b.el.style.top = `${Math.max(r.top + window.scrollY + 8, window.scrollY + 4)}px`;
  }
  let queued = false;
  const placeAll = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; badges.forEach(place); }); };
  window.addEventListener("scroll", placeAll, { passive: true });
  window.addEventListener("resize", placeAll);

  function body(msg) {
    const top = `<div class="top">${MARK}<span class="brand">Deep-Guard Lens</span><button class="x" title="Close" aria-label="Close">×</button></div>`;
    if (msg.phase === "checking") return { tone: "", html: `${top}<div class="line"><span class="spin"></span>Checking this picture…</div>` };
    if (msg.phase === "error") return { tone: "err", html: `${top}<div class="head">Couldn't check it</div><div class="line">${esc(msg.error)}</div>` };
    const r = msg.result;
    const v = VERDICT[r.verdict] || { label: r.headline, tone: "" };
    const lines = [];
    if (r.match) {
      const how = r.match.kind === "exact" ? (r.match.exact_file === "credentialed_copy" ? "the credentialed copy itself" : "the exact file")
        : r.match.kind === "watermark" ? "hidden ID read from the pixels"
        : r.match.kind === "visual" ? "looks the same" : "same content";
      lines.push(`“${esc(r.match.title)}” by ${esc(r.match.creator)}${r.match.block !== null && r.match.block !== undefined ? ` · block #${esc(r.match.block)}` : " · pending"}`);
      lines.push(`<span class="k">How:</span> ${esc(how)}`);
    }
    if (r.hidden_id) lines.push(`<span class="k">Hidden ID:</span> <span class="mono">${esc(r.hidden_id)}</span>`);
    lines.push(`<span class="cr">cr</span>${esc(CR_TEXT[r.credentials.verdict] || r.credentials.verdict)}${r.credentials.ai_declared ? " · declares AI" : ""}`);
    if (r.detector) {
      lines.push(`<span class="k">AI detector:</span> ${Math.round(r.detector.raw * 100)}% AI <span class="k">— a guess, not proof</span>`);
    }
    for (const n of (r.notes || []).slice(0, 2)) lines.push(`<span class="k">${esc(n)}</span>`);
    const links = [r.links.record ? `<a href="${esc(r.links.record)}" target="_blank" rel="noopener">See the proof</a>` : "",
                   `<a href="${esc(r.links.ledger)}" target="_blank" rel="noopener">Ledger</a>`].filter(Boolean).join("");
    return { tone: v.tone, html: `${top}<div class="head">${r.verdict} · ${esc(v.label)}</div>${lines.map((l) => `<div class="line">${l}</div>`).join("")}<div class="links">${links}</div>` };
  }

  function show(msg) {
    let b = badges.get(msg.id);
    if (!b) {
      const el = document.createElement("div");
      el.setAttribute("role", "status");
      root.appendChild(el);
      b = { el, img: findImage(msg.srcUrl) };
      badges.set(msg.id, b);
      // A newer check of the same picture replaces the older badge.
      for (const [id, other] of badges) if (id !== msg.id && other.img && other.img === b.img) { other.el.remove(); badges.delete(id); }
    }
    const { tone, html } = body(msg);
    b.el.className = `b ${tone}`;
    b.el.innerHTML = html;
    b.el.querySelector(".x").addEventListener("click", () => { b.el.remove(); badges.delete(msg.id); });
    place(b);
  }

  function visible(img) {
    const r = img.getBoundingClientRect();
    return r.width >= 120 && r.height >= 120 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
  }

  chrome.runtime.onMessage.addListener((msg, sender, reply) => {
    if (msg && msg.type === "dg-lens") show(msg);
    if (msg && msg.type === "dg-list-images") {
      const seen = new Set();
      reply([...document.images].filter((i) => i.complete && i.naturalWidth >= 120 && visible(i))
        .map((i) => i.currentSrc || i.src).filter((s) => s && !seen.has(s) && seen.add(s)).slice(0, 8));
    }
    return false;
  });
})();
