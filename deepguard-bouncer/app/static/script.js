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

const statusDot  = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

let selectedFile = null;

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
  } catch {
    statusDot.className = "status__dot is-bad";
    statusText.textContent = "Service offline";
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

function render(data) {
  const isVideo = data.input_type === "video";
  const subject = isVideo ? "video" : "image";
  const fakePct = data.probability_fake * 100;

  let verdict, headline, explain;

  // Videos get an extra verdict that images can't have: manipulation
  // confined to PART of a clip. A localised face swap can leave the
  // overall average low while a handful of frames score near-certain,
  // so we check the flagged fraction before trusting the average.
  const partial = isVideo
    && data.flagged_ratio >= 0.12
    && data.peak_probability >= 0.75
    && fakePct < 65;

  if (partial) {
    verdict  = "manipulated";
    headline = "Part of this video looks AI-generated.";
    explain  = `${data.frames_flagged} of the ${data.frames_analysed} frames checked show strong signs of manipulation, `
             + `even though the clip as a whole averages lower. Check the timeline below for where.`;
  } else if (fakePct >= 65) {
    verdict  = "manipulated";
    headline = isVideo ? "This video looks AI-generated." : "This looks AI-generated.";
    explain  = `The model estimates a ${Math.round(fakePct)}% chance that this ${subject} was produced or altered by AI.`;
  } else if (fakePct <= 35) {
    verdict  = "authentic";
    headline = isVideo ? "This video looks authentic." : "This looks authentic.";
    explain  = `The model estimates a ${Math.round(100 - fakePct)}% chance that this ${subject} is genuine and unmanipulated.`;
  } else {
    verdict  = "uncertain";
    headline = "The result is inconclusive.";
    explain  = `At ${Math.round(fakePct)}% the reading sits too close to the middle to call either way. `
             + `Try a clearer, higher-resolution ${isVideo ? "clip" : "photo"} showing a face.`;
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
