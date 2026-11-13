// Deep-Guard Lens popup: which server to ask, whether it answers, the
// "check this page" button and the last few checks.
"use strict";

const DEFAULT_SERVER = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const VERDICT = { 1: "Authentic & unaltered", 2: "Same picture, file modified", 3: "No provenance found", 4: "Declared AI-generated" };

function setStatus(el, text, tone = "") { el.textContent = text; el.className = `status ${tone}`; }

async function testConnection(base) {
  setStatus($("status"), "Checking the connection…");
  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 6000);
    const r = await fetch(`${base}/api/ledger/summary`, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    setStatus($("status"), `Connected · ledger has ${s.block_count} blocks, ${s.record_count} records`, "ok");
  } catch (e) {
    setStatus($("status"), `No Deep-Guard server answered at ${base}. Start it with python run_deepguard.py.`, "bad");
  }
}

async function loadRecent() {
  const { recent = [] } = await chrome.storage.session.get("recent");
  if (!recent.length) return;
  $("recent").innerHTML = recent.map((r) => `<li class="v${esc(r.verdict)}"><b>${esc(r.verdict)} · ${esc(VERDICT[r.verdict] || r.headline)}</b>
    ${r.title ? `<small>“${esc(r.title)}”</small>` : ""}<small>${esc(r.srcUrl)}</small></li>`).join("");
}

$("serverForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const value = $("server").value.trim().replace(/\/+$/, "") || DEFAULT_SERVER;
  await chrome.storage.local.set({ server: value });
  $("server").value = value;
  testConnection(value);
});

$("scanBtn").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  $("scanBtn").disabled = true;
  setStatus($("scanOut"), "Checking the visible pictures… the badges appear on the page.");
  const d = await chrome.runtime.sendMessage({ type: "dg-scan-page", tabId: tab.id });
  $("scanBtn").disabled = false;
  if (d.error) setStatus($("scanOut"), d.error, "bad");
  else setStatus($("scanOut"), d.checked ? `Checked ${d.checked} picture${d.checked === 1 ? "" : "s"}${d.failed ? `, ${d.failed} failed` : ""}.` : "No large enough picture is visible on this page.", d.failed ? "bad" : "ok");
  loadRecent();
});

(async () => {
  const { server } = await chrome.storage.local.get("server");
  $("server").value = server || DEFAULT_SERVER;
  testConnection(server || DEFAULT_SERVER);
  loadRecent();
})();
