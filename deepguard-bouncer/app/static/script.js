// Deep-Guard — interface behaviour.
// Plain JS, no build step. Served directly by FastAPI from /static.

const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("fileInput");
const dropzoneEmpty  = document.getElementById("dropzoneEmpty");
const previewWrap    = document.getElementById("previewWrap");
const previewImg     = document.getElementById("previewImg");
const previewVideo   = document.getElementById("previewVideo");
const fileNameEl     = document.getElementById("fileName");
const fileNoteEl     = document.getElementById("fileNote");
const changeBtn      = document.getElementById("changeBtn");
const runBtn         = document.getElementById("runBtn");
const errorMsg       = document.getElementById("errorMsg");

const result          = document.getElementById("result");
const verdictHeadline = document.getElementById("verdictHeadline");
const gaugeFill       = document.getElementById("gaugeFill");
const readoutFigure   = document.getElementById("readoutFigure");
const readoutExplain  = document.getElementById("readoutExplain");
const verdictLegend   = document.getElementById("verdictLegend");
const timelineBlock   = document.getElementById("timelineBlock");
const timelineIntro   = document.getElementById("timelineIntro");
const timelineChart   = document.getElementById("timelineChart");
const statsBlock      = document.getElementById("statsBlock");
const axisStart       = document.getElementById("axisStart");
const axisEnd         = document.getElementById("axisEnd");

const explainBlock    = document.getElementById("explainBlock");
const heatmapImg      = document.getElementById("heatmapImg");
const consistencyBlock = document.getElementById("consistencyBlock");

const frequencyBlock       = document.getElementById("frequencyBlock");
const frequencySpectrumImg = document.getElementById("frequencySpectrumImg");
const frequencyCaption     = document.getElementById("frequencyCaption");
const frequencyStats       = document.getElementById("frequencyStats");

const shapBtn           = document.getElementById("shapBtn");
const shapHint          = document.getElementById("shapHint");
const shapRegionsBlock  = document.getElementById("shapRegionsBlock");
const shapRegionsIntro  = document.getElementById("shapRegionsIntro");
const shapRegionsList   = document.getElementById("shapRegionsList");

const attributionBlock   = document.getElementById("attributionBlock");
const attributionIntro   = document.getElementById("attributionIntro");
const attributionBars    = document.getElementById("attributionBars");

const fingerprintBlock  = document.getElementById("fingerprintBlock");
const fingerprintFields = document.getElementById("fingerprintFields");
const compressionTestBtn   = document.getElementById("compressionTestBtn");
const compressionTestHint  = document.getElementById("compressionTestHint");
const compressionTestBlock = document.getElementById("compressionTestBlock");
const compressionTestIntro = document.getElementById("compressionTestIntro");
const compressionTestTable = document.getElementById("compressionTestTable");

const verifyBtn  = document.getElementById("verifyBtn");
const verifyHint = document.getElementById("verifyHint");

const statusDot  = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const signerPublicKey = document.getElementById("signerPublicKey");

let selectedFile = null;
let currentFingerprintRecord = null;

/* ------------------------------------------------------------- health -- */

(async function checkHealth() {
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      statusDot.className = "status__dot is-ok";
      statusText.textContent = "Ready";
    } else {
      statusDot.className = "status__dot is-bad";
      statusText.textContent = "Model unavailable";
    }
    signerPublicKey.textContent = data.signer_public_key || "Not available — signing key failed to load.";
  } catch {
    statusDot.className = "status__dot is-bad";
    statusText.textContent = "Service offline";
    signerPublicKey.textContent = "Not available — service offline.";
  }
})();

/* ---------------------------------------------------------- selection -- */

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function selectFile(file) {
  if (!file) return;

  hideError();
  result.hidden = true;
  selectedFile = file;

  const isVideo = file.type.startsWith("video/");
  const url = URL.createObjectURL(file);

  if (isVideo) {
    previewVideo.src = url;
    previewVideo.hidden = false;
    previewImg.hidden = true;
  } else {
    previewImg.src = url;
    previewImg.hidden = false;
    previewVideo.hidden = true;
  }

  fileNameEl.textContent = file.name;
  // Be explicit about how videos are handled — otherwise a user has no
  // idea the whole clip isn't being analysed frame by frame.
  fileNoteEl.textContent = isVideo
    ? formatSize(file.size) + " · frames sampled across the whole clip will be analysed"
    : formatSize(file.size) + " · image";

  dropzoneEmpty.hidden = true;
  previewWrap.hidden = false;

  runBtn.disabled = false;
  runBtn.textContent = isVideo ? "Analyse this video" : "Analyse this image";
}

fileInput.addEventListener("change", e => selectFile(e.target.files[0]));

changeBtn.addEventListener("click", e => {
  e.preventDefault();
  e.stopPropagation();   // don't let the click bubble to the <label>
  fileInput.click();
});

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  })
);

["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);

dropzone.addEventListener("drop", e => selectFile(e.dataTransfer.files[0]));

/* -------------------------------------------------------------- errors -- */

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}

/* ------------------------------------------------------------ rendering -- */

function animateFigure(target) {
  const start = performance.now();
  const duration = 620;
  (function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    readoutFigure.textContent = Math.round(target * eased) + "%";
    if (p < 1) requestAnimationFrame(step);
  })(start);
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m + ":" + String(s).padStart(2, "0");
}

function renderTimeline(data) {
  const frames = data.timeline || [];
  timelineChart.innerHTML = "";

  frames.forEach(f => {
    const pct = f.p * 100;
    const bar = document.createElement("div");
    bar.className = "bar" + (pct >= 65 ? " is-flagged" : pct >= 35 ? " is-mid" : "");
    bar.style.height = Math.max(pct, 2) + "%";
    bar.dataset.label = formatTime(f.t) + " — " + Math.round(pct) + "% AI";
    timelineChart.appendChild(bar);
  });

  const dur = data.video?.duration_seconds || 0;
  axisStart.textContent = "0:00";
  axisEnd.textContent = dur ? formatTime(dur) : "end";

  timelineIntro.textContent =
    `Each bar is one sampled frame, in order from the start of the clip to the end. `
    + `Bar height is how likely that frame is AI-generated. `
    + `${data.frames_analysed} frames were checked across ${dur ? formatTime(dur) : "the clip"}.`;

  const stats = [
    ["Frames analysed", data.frames_analysed],
    ["Frames flagged",  `${data.frames_flagged} of ${data.frames_analysed}`],
    ["Average score",   Math.round(data.mean_probability * 100) + "%"],
    ["Highest frame",   Math.round(data.peak_probability * 100) + "%"],
    ["Resolution",      data.video?.resolution || "—"],
    ["Analysis time",   data.elapsed_seconds + "s"],
  ];

  statsBlock.innerHTML = stats.map(([label, value]) =>
    `<div><dt>${label}</dt><dd>${value}</dd></div>`
  ).join("");

  timelineBlock.hidden = false;
}

function renderExplainability(data) {
  const exp = data.explainability;

  // A fresh analysis invalidates any SHAP breakdown left over from a
  // previous image — it's opt-in and tied to whichever file is
  // currently selected, so it must not silently persist across runs.
  shapRegionsBlock.hidden = true;
  shapHint.hidden = true;
  shapRegionsList.innerHTML = "";

  if (!exp || !exp.heatmap_png_base64) {
    explainBlock.hidden = true;
    return;
  }

  heatmapImg.src = "data:image/png;base64," + exp.heatmap_png_base64;

  const c = exp.model_consistency || {};
  const rows = [
    ["Repeat-check average", c.mean_probability != null ? Math.round(c.mean_probability * 100) + "%" : "—"],
    ["Wobble across re-checks", c.std_probability != null ? c.std_probability.toFixed(3) : "—"],
  ];
  if (exp.frame_timestamp != null) {
    rows.unshift(["Frame shown", formatTime(exp.frame_timestamp)]);
  }

  consistencyBlock.innerHTML = rows.map(([label, value]) =>
    `<div><dt>${label}</dt><dd>${value}</dd></div>`
  ).join("") + (c.note ? `<div class="note">${c.note}</div>` : "");

  // Region attribution only supports still images — the endpoint
  // re-scores masked crops of a single frame, which isn't meaningful
  // averaged across a video's sampled frames.
  shapBtn.hidden = data.input_type === "video";

  renderFrequencyPanel(exp.frequency_analysis);

  explainBlock.hidden = false;
}

async function runShapRegions() {
  if (!selectedFile) return;

  shapBtn.disabled = true;
  shapBtn.textContent = "Computing… (a few seconds)";
  shapHint.hidden = true;

  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const res = await fetch("/api/explain/shap-regions", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned an error (${res.status}).`);
    }
    const data = await res.json();

    shapRegionsIntro.textContent =
      `Each region's fair share of credit for this verdict (Shapley values), estimated by `
      + `re-scoring the image with regions masked out. Top ${data.regions.length} of `
      + `${data.segments_total} regions shown, strongest first.`;

    shapRegionsList.innerHTML = data.regions.map(r => {
      const towardFake = r.contribution >= 0;
      return `<div class="shap-region">
        <img class="shap-region__thumb" src="data:image/png;base64,${r.thumbnail_png_base64}" alt="Image region" />
        <div>
          <div class="shap-region__value ${towardFake ? "toward-fake" : "toward-real"}">
            ${towardFake ? "+" : ""}${r.contribution.toFixed(3)}
          </div>
          <div class="shap-region__label">toward ${towardFake ? "AI-generated" : "real"}</div>
        </div>
      </div>`;
    }).join("");

    shapRegionsBlock.hidden = false;
  } catch (err) {
    shapHint.textContent = err.message || "Something went wrong computing the region breakdown.";
    shapHint.hidden = false;
  } finally {
    shapBtn.disabled = false;
    shapBtn.textContent = "Show region-by-region contribution →";
  }
}

shapBtn.addEventListener("click", runShapRegions);

// Generic title-casing gets most of these right (stable_diffusion ->
// "Stable Diffusion") but mangles proper stylizations for acronym-style
// names -- caught by actually looking at the rendered page, not assumed.
const GENERATOR_DISPLAY_NAMES = {
  ddpm: "DDPM",
  stylegan2: "StyleGAN2",
  pro_gan: "ProGAN",
  big_gan: "BigGAN",
  cycle_gan: "CycleGAN",
};

function formatGeneratorName(slug) {
  return GENERATOR_DISPLAY_NAMES[slug]
    || slug.split("_").map(w => w[0].toUpperCase() + w.slice(1)).join(" ");
}

function renderAttribution(data, displayedVerdict) {
  // Deliberately checks the VERDICT ACTUALLY SHOWN as the headline
  // (displayedVerdict, computed in render() below), not the server's own
  // raw data.verdict field. The server flags "manipulated" at a flat 50%,
  // but the headline uses a wider uncertain-zone (see render()'s own
  // comment on MANIPULATED_THRESHOLD) -- a score in that gap would
  // otherwise show "The result is inconclusive" as the headline while
  // this panel confidently named a specific generator underneath it,
  // which is a real, confusing contradiction, not just a style nit.
  const attr = data.generator_attribution;
  if (displayedVerdict !== "manipulated" || !attr) {
    attributionBlock.hidden = true;
    return;
  }

  attributionIntro.textContent =
    `Once an image is flagged as AI-generated, this model (trained separately from the detector above) `
    + `estimates which specific tool most likely made it.`;

  const entries = Object.entries(attr.probabilities); // already sorted highest-first by the server
  attributionBars.innerHTML = entries.map(([name, prob], i) => `
    <div class="attribution__bar-row ${i === 0 ? "is-top" : ""}">
      <div class="attribution__bar-label">${formatGeneratorName(name)}</div>
      <div class="attribution__bar-track"><div class="attribution__bar-fill" style="width: ${Math.round(prob * 100)}%"></div></div>
      <div class="attribution__bar-value">${Math.round(prob * 100)}%</div>
    </div>
  `).join("");

  attributionBlock.hidden = false;
}

function renderFrequencyPanel(freq) {
  if (!freq || !freq.spectrum_png_base64) {
    frequencyBlock.hidden = true;
    return;
  }

  frequencySpectrumImg.src = "data:image/png;base64," + freq.spectrum_png_base64;

  // Deliberately understated: this heuristic did NOT reliably separate
  // real from AI images in our own testing (see frequency_analysis.py),
  // so the copy here describes what's shown rather than asserting a
  // finding the evidence doesn't support.
  frequencyCaption.textContent =
    `The image's frequency spectrum, computed directly from its pixels — independent of the `
    + `model above. Center = overall brightness; edges = fine detail. Real cameras and AI `
    + `generators can leave different traces here, but this is an experimental signal, not a `
    + `validated detector — read the shape of the plot as illustrative, not as evidence on its own.`;

  frequencyStats.innerHTML = [
    ["Statistical outliers flagged", String(freq.peak_count)],
    ["Anomaly score", freq.anomaly_score.toFixed(3)],
  ].map(([label, value]) =>
    `<div><dt>${label}</dt><dd>${value}</dd></div>`
  ).join("");

  frequencyBlock.hidden = false;
}

function truncateHash(hex, keep = 12) {
  if (!hex || hex.length <= keep * 2 + 3) return hex;
  return hex.slice(0, keep) + " … " + hex.slice(-keep);
}

function renderFingerprint(data) {
  const fp = data.fingerprint;
  compressionTestBlock.hidden = true;
  compressionTestHint.hidden = true;
  verifyHint.hidden = true;
  currentFingerprintRecord = fp || null;

  if (!fp) {
    fingerprintBlock.hidden = true;
    return;
  }

  const isVideo = data.input_type === "video";
  const rows = [
    ["Exact fingerprint (SHA-256)", fp.sha256_merkle_root],
    [isVideo ? "Visual fingerprint (per sampled frame)" : "Visual fingerprint (perceptual hash)",
      isVideo ? `${fp.phash_frames?.length ?? 0} frame hashes computed` : fp.phash],
    ["Signed at", fp.timestamp_utc ? new Date(fp.timestamp_utc).toLocaleString() : "—"],
    ["Signature", fp.signature],
  ];

  fingerprintFields.innerHTML = rows.map(([label, value]) =>
    `<div><dt>${label}</dt><dd>${value ?? "—"}</dd></div>`
  ).join("") + `<div><dt>Status</dt><dd class="is-signed">&#10003; Signed with this server's Ed25519 key</dd></div>`;

  // Compression-resilience test only makes sense for images — the
  // endpoint re-saves a single image at several JPEG qualities.
  compressionTestBtn.hidden = isVideo;
  fingerprintBlock.hidden = false;
}

async function verifyRecord(record) {
  const res = await fetch("/api/fingerprint/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server returned an error (${res.status}).`);
  }
  return (await res.json()).valid;
}

async function runVerify() {
  if (!currentFingerprintRecord) return;

  verifyBtn.disabled = true;
  verifyBtn.textContent = "Verifying…";
  verifyHint.hidden = true;

  try {
    const genuineValid = await verifyRecord(currentFingerprintRecord);

    // Prove the check isn't vacuous, the same way the compression test
    // proves the exact hash actually changes: flip one character of the
    // hash and confirm verification now correctly FAILS.
    const tampered = { ...currentFingerprintRecord };
    tampered.sha256_merkle_root = (tampered.sha256_merkle_root[0] === "f" ? "e" : "f")
      + tampered.sha256_merkle_root.slice(1);
    const tamperedValid = await verifyRecord(tampered);

    const success = genuineValid && !tamperedValid;
    verifyHint.classList.toggle("is-success", success);
    verifyHint.textContent = success
      ? "✓ Verified — this exact record was signed by the key above. A tampered copy (one hash character flipped) correctly fails the same check."
      : `Unexpected result (genuine valid=${genuineValid}, tampered valid=${tamperedValid}) — see server logs.`;
    verifyHint.hidden = false;
  } catch (err) {
    verifyHint.classList.remove("is-success");
    verifyHint.textContent = err.message || "Something went wrong verifying that record.";
    verifyHint.hidden = false;
  } finally {
    verifyBtn.disabled = false;
    verifyBtn.textContent = "Verify independently →";
  }
}

verifyBtn.addEventListener("click", runVerify);

async function runCompressionTest() {
  if (!selectedFile) return;

  compressionTestBtn.disabled = true;
  compressionTestBtn.textContent = "Testing…";
  compressionTestHint.hidden = true;

  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const res = await fetch("/api/fingerprint/compression-test", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned an error (${res.status}).`);
    }
    const data = await res.json();

    compressionTestIntro.textContent =
      `Same image, re-saved at decreasing JPEG quality (simulating WhatsApp/Twitter-style `
      + `compression). The exact fingerprint should change completely every time; the visual `
      + `fingerprint should barely move.`;

    const headRow = `<tr><th>Quality</th><th>File size</th><th>Exact fingerprint</th><th>Visual fingerprint drift</th></tr>`;
    const bodyRows = data.recompressed.map(r => {
      const pct = Math.round((r.file_size_bytes / data.original.file_size_bytes) * 100);
      return `<tr>
        <td>${r.quality}</td>
        <td>${formatSize(r.file_size_bytes)} (${pct}% of original)</td>
        <td class="mono sha-changed">${truncateHash(r.sha256_merkle_root)}</td>
        <td class="phash-close">${r.phash_bits_different} / ${data.phash_bits_total} bits different</td>
      </tr>`;
    }).join("");

    compressionTestTable.innerHTML = headRow + bodyRows;
    compressionTestBlock.hidden = false;
  } catch (err) {
    compressionTestHint.textContent = err.message || "Something went wrong running the compression test.";
    compressionTestHint.hidden = false;
  } finally {
    compressionTestBtn.disabled = false;
    compressionTestBtn.textContent = "Test compression resilience →";
  }
}

compressionTestBtn.addEventListener("click", runCompressionTest);

function render(data) {
  const isVideo = data.input_type === "video";
  const subject = isVideo ? "video" : "image";
  const fakePct = data.probability_fake * 100;

  let verdict, headline, explain;

  // Videos get an extra verdict that images can't have: manipulation
  // confined to PART of a clip. A localised face swap can leave the
  // overall average low while a handful of frames score near-certain,
  // so we check the flagged fraction before trusting the average.
  //
  // The 80/20 split below (rather than a plain 50/50) was originally added
  // to compensate for the FIRST deployed model's bias: overconfidence on
  // single-face portraits, which it turned real photos into "manipulated"
  // verdicts in the high 90s for. That model has since been swapped for a
  // retrained one with a DIFFERENT, roughly opposite bias (documented in
  // the verdict__caveat text below): it can miss certain AI-generated
  // images entirely, scoring them confidently "authentic" in the low
  // single digits. Widening the uncertain zone around 50% does nothing for
  // that failure mode -- a 4% score is nowhere near the midpoint no matter
  // how wide the band is -- so this logic is kept as a generally reasonable
  // conservative default, not because it targets the current model's
  // actual weak spot. There currently isn't a UI-level mitigation for that
  // one; it needs a better model (Generator Attribution / a future
  // retrain), not a smarter threshold.
  const partial = isVideo
    && data.flagged_ratio >= 0.12
    && data.peak_probability >= 0.75
    && fakePct < 80;

  if (partial) {
    verdict  = "manipulated";
    headline = "Part of this video looks AI-generated.";
    explain  = `${data.frames_flagged} of the ${data.frames_analysed} frames checked show strong signs of manipulation, `
             + `even though the clip as a whole averages lower. Check the timeline below for where.`;
  } else if (fakePct >= 80) {
    verdict  = "manipulated";
    headline = isVideo ? "This video looks AI-generated." : "This looks AI-generated.";
    explain  = `The model estimates a ${Math.round(fakePct)}% chance that this ${subject} was produced or altered by AI.`;
  } else if (fakePct <= 20) {
    verdict  = "authentic";
    headline = isVideo ? "This video looks authentic." : "This looks authentic.";
    explain  = `The model estimates a ${Math.round(100 - fakePct)}% chance that this ${subject} is genuine and unmanipulated.`;
  } else {
    verdict  = "uncertain";
    headline = "The result is inconclusive.";
    explain  = `At ${Math.round(fakePct)}% the reading sits too close to the middle to call either way. `
             + `Try a clearer, higher-resolution ${isVideo ? "clip" : "photo"}.`;
  }

  const displayPct = verdict === "authentic" ? 100 - fakePct : fakePct;

  result.dataset.verdict = verdict;
  result.hidden = false;
  verdictHeadline.textContent = headline;
  readoutExplain.textContent = explain;
  animateFigure(Math.round(displayPct));

  // Marker position always maps to P(AI): left = real, right = AI.
  requestAnimationFrame(() => { gaugeFill.style.left = fakePct + "%"; });

  verdictLegend.textContent =
    `What this number means: it is the model's confidence, from 0% to 100%, that this ${subject} is `
    + (verdict === "authentic" ? "genuine rather than AI-generated." : "AI-generated rather than genuine.")
    + ` The marker above shows where this result falls between "definitely real" and "definitely AI".`;

  if (isVideo && data.timeline?.length) {
    renderTimeline(data);
  } else {
    timelineBlock.hidden = true;
  }

  renderExplainability(data);
  renderAttribution(data, verdict);
  renderFingerprint(data);

  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* --------------------------------------------------------------- submit -- */

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  result.hidden = true;
  runBtn.disabled = true;
  runBtn.textContent = selectedFile.type.startsWith("video/") ? "Analysing frames…" : "Analysing…";
  dropzone.classList.add("is-busy");

  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `The server returned an error (${res.status}).`);
    }
    render(await res.json());
  } catch (err) {
    showError(
      String(err.message).includes("Failed to fetch")
        ? "Couldn't reach the analysis service. Check that the server is still running, then try again."
        : err.message || "Something went wrong while analysing that file."
    );
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = selectedFile && selectedFile.type.startsWith("video/")
      ? "Analyse this video" : "Analyse this image";
    dropzone.classList.remove("is-busy");
  }
});
