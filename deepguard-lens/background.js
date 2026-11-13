// Deep-Guard Lens — the extension's background service worker.
//
// Right-click an image -> "Check with Deep-Guard": this worker downloads the
// image itself (it may, for any site: host_permissions), sends it to the
// Deep-Guard server the user chose (local or the public link) at
// /api/ledger/verify, and hands a short summary to content.js, which pins a
// badge on the picture. Everything the badge says comes from the server's
// answer; the extension decides nothing itself.
"use strict";

const DEFAULT_SERVER = "http://127.0.0.1:8000";
const RECENT_LIMIT = 12;

async function serverUrl() {
  const { server } = await chrome.storage.local.get("server");
  return String(server || DEFAULT_SERVER).trim().replace(/\/+$/, "");
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "dg-check", title: "Check with Deep-Guard", contexts: ["image"] });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "dg-check" && tab && info.srcUrl) checkImage(tab.id, info.srcUrl, info.frameId || 0);
});

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg && msg.type === "dg-scan-page") {
    scanPage(msg.tabId).then(reply, (e) => reply({ error: String(e.message || e) }));
    return true; // the answer comes later
  }
  return false;
});

async function ensureContent(tabId, frameId) {
  await chrome.scripting.executeScript({ target: { tabId, frameIds: [frameId] }, files: ["content.js"] });
}

function send(tabId, frameId, message) {
  return chrome.tabs.sendMessage(tabId, message, { frameId }).catch(() => { /* the tab went away */ });
}

const EXT = { "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif", "image/bmp": ".bmp", "image/avif": ".avif" };

function nameFor(srcUrl, type) {
  let base = "image";
  try { base = decodeURIComponent(new URL(srcUrl).pathname.split("/").pop() || "image"); } catch (e) { /* data: or odd URL */ }
  base = base.replace(/[^\w.\-]+/g, "_").slice(0, 80) || "image";
  return /\.(jpe?g|png|webp|bmp|gif|avif)$/i.test(base) ? base : base + (EXT[type] || ".jpg");
}

// The image's bytes. A blob: URL only exists inside the page that made it,
// so that one is read by a tiny function run in the page.
async function imageBlob(tabId, frameId, srcUrl) {
  if (srcUrl.startsWith("blob:")) {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId, frameIds: [frameId] },
      func: async (url) => {
        const b = await (await fetch(url)).blob();
        return await new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(b); });
      },
      args: [srcUrl],
    });
    srcUrl = result;
  }
  let response = await fetch(srcUrl, { credentials: "include" }).catch(() => null);
  if (!response || !response.ok) response = await fetch(srcUrl);  // some sites refuse credentialed requests
  if (!response.ok) throw new Error(`the image could not be downloaded (HTTP ${response.status})`);
  return response.blob();
}

async function errorText(response) {
  try {
    const body = await response.json();
    const d = body.detail;
    return typeof d === "string" ? d : (d && d.message) || JSON.stringify(d || body);
  } catch (e) { return `HTTP ${response.status}`; }
}

// The badge's view of the server's answer: small, and free of server internals.
function summarise(d, base) {
  const m = d.match;
  const c = d.credentials || {};
  return {
    verdict: d.verdict_number,
    headline: d.headline,
    match: m ? {
      title: m.record.title, creator: m.record.creator_handle, block: m.block_index, kind: m.kind,
      exact_file: m.exact_file, declared_origin: m.record.declared_origin, record_id: m.record.record_id,
      phash_bits: m.phash_bits_different, similarity: m.content_similarity,
    } : null,
    hidden_id: d.watermark && d.watermark.matched ? d.watermark.id : null,
    credentials: { verdict: c.verdict || "none", headline: c.headline || "", signer: c.signer || null,
                   ai_declared: !!c.ai_declared, source: c.source_type_text || null },
    detector: d.detector ? { raw: d.detector.probability_fake, calibrated: d.detector.calibrated_probability_fake } : null,
    notes: d.notes || [],
    links: { record: m ? `${base}/explorer#record=${m.record.record_id}` : null, ledger: `${base}/ledger` },
  };
}

async function remember(entry) {
  const { recent = [] } = await chrome.storage.session.get("recent");
  recent.unshift(entry);
  await chrome.storage.session.set({ recent: recent.slice(0, RECENT_LIMIT) });
}

async function checkImage(tabId, srcUrl, frameId = 0) {
  const id = crypto.randomUUID();
  try { await ensureContent(tabId, frameId); } catch (e) { return { error: "This page doesn't allow extensions (a browser page or the web store)." }; }
  await send(tabId, frameId, { type: "dg-lens", phase: "checking", id, srcUrl });
  const base = await serverUrl();
  try {
    const blob = await imageBlob(tabId, frameId, srcUrl);
    const form = new FormData();
    form.append("file", blob, nameFor(srcUrl, blob.type));
    form.append("ai_fallback", "true");
    let response;
    try {
      response = await fetch(`${base}/api/ledger/verify`, { method: "POST", body: form });
    } catch (e) {
      throw new Error(`no Deep-Guard server answered at ${base}. Start it (python run_deepguard.py) or set its address in the Lens popup.`);
    }
    if (!response.ok) throw new Error(await errorText(response));
    const result = summarise(await response.json(), base);
    await send(tabId, frameId, { type: "dg-lens", phase: "result", id, srcUrl, result });
    const tab = await chrome.tabs.get(tabId).catch(() => ({}));
    await remember({ at: Date.now(), page: tab.url || "", srcUrl: srcUrl.startsWith("data:") ? "(embedded image)" : srcUrl,
                     verdict: result.verdict, headline: result.headline, title: result.match && result.match.title,
                     credentials: result.credentials.verdict });
    return result;
  } catch (e) {
    const error = String(e.message || e);
    await send(tabId, frameId, { type: "dg-lens", phase: "error", id, srcUrl, error });
    return { error };
  }
}

// The popup's "check the images on this page": the visible, reasonably
// large images, one after another (the server checks one at a time anyway).
async function scanPage(tabId) {
  await ensureContent(tabId, 0);
  const images = await chrome.tabs.sendMessage(tabId, { type: "dg-list-images" }, { frameId: 0 });
  const results = [];
  for (const srcUrl of images || []) results.push(await checkImage(tabId, srcUrl, 0));
  return { checked: results.length, failed: results.filter((r) => r.error).length };
}

self.dgLens = { checkImage, scanPage, serverUrl, summarise };  // for the tests (and the curious)
