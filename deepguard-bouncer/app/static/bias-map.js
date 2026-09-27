// Deep-Guard — Bias Map page behaviour. Plain JS, no build step, same as script.js.
//
// bias_map/results.json is written by evaluation/run_exam.py --write-bias-map
// from the exam set. It carries a list of `sections`, each a set of groups
// (a generator, a kind of real photo, a subject, an image size...) with how
// often the analysis page's verdict was right, wrong, or inconclusive.

// Defense in depth, same reasoning as registry.js's escapeHtml: group names
// come from dataset metadata and the exam manifest, which today need
// filesystem access to edit -- lower risk than registry.js's remotely-
// writable title, but "currently harder to reach" isn't a reason to render
// them unescaped.
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const pct = x => (x === null || x === undefined) ? "—" : Math.round(x * 100) + "%";

function accuracyClass(entry) {
  if (!entry.trustworthy) return "is-unsure";
  if (entry.accuracy >= 0.8) return "is-good";
  if (entry.accuracy >= 0.5) return "is-mid";
  return "is-bad";
}

function cellDetail(entry) {
  // Older results files have only accuracy and a count.
  if (entry.wrong === undefined) return "";
  return `<div class="bias-map__cell-detail">${pct(entry.wrong)} wrong · ${pct(entry.inconclusive)} unsure</div>`;
}

function renderGrid(groups) {
  const entries = Object.entries(groups).sort((a, b) => a[1].accuracy - b[1].accuracy);
  return entries.map(([label, entry]) => `
    <div class="bias-map__cell ${accuracyClass(entry)}">
      <div class="bias-map__cell-label">${escapeHtml(label)}</div>
      <div class="bias-map__cell-value">${pct(entry.accuracy)}</div>
      ${cellDetail(entry)}
      <div class="bias-map__cell-n">n=${entry.sample_count}${entry.trustworthy ? "" : " (too small to trust)"}</div>
    </div>
  `).join("");
}

function renderSection(title, note, groups) {
  return `
    <section class="bias-map__section">
      <h2 class="bias-map__section-title">${escapeHtml(title)}</h2>
      ${note ? `<p class="bias-map__section-note">${escapeHtml(note)}</p>` : ""}
      <div class="bias-map__grid">${renderGrid(groups)}</div>
    </section>`;
}

function stat(value, label) {
  return `<div class="bias-map__stat">
      <div class="bias-map__stat-value">${escapeHtml(value)}</div>
      <div class="bias-map__stat-label">${escapeHtml(label)}</div>
    </div>`;
}

(async function load() {
  const scopeWarning = document.getElementById("scopeWarning");
  const scopeNote    = document.getElementById("scopeNote");
  const overallBlock = document.getElementById("overallBlock");
  const sections     = document.getElementById("sections");
  const emptyBlock   = document.getElementById("emptyBlock");

  try {
    const res = await fetch("/api/bias-map");
    const data = await res.json();

    if (!data.generated) {
      emptyBlock.hidden = false;
      return;
    }

    if (data.scope_warning) {
      scopeWarning.textContent = data.scope_warning;
      scopeWarning.hidden = false;
    }
    if (data.scope_note) {
      scopeNote.textContent = data.scope_note;
      scopeNote.hidden = false;
    }

    const when = new Date(data.generated_at_utc).toLocaleDateString();
    const h = data.headline;
    overallBlock.innerHTML = h
      ? stat(pct(h.caught), "AI images called AI-generated")
        + stat(pct(h.false_alarm), "Real photos called AI-generated")
        + stat(String(data.total_images), "Images in the exam")
        + stat(when, "Last run")
      : stat(pct(data.overall_accuracy), "Overall accuracy")
        + stat(String(data.total_images), "Images in this run")
        + stat(when, "Last generated");
    overallBlock.hidden = false;

    const list = data.sections || [
      { title: "By subject", groups: data.by_subject },
      { title: "By style", groups: data.by_style },
      { title: "By generator", groups: data.by_generator },
      { title: "By generator × subject × style", groups: data.by_combination },
    ];
    sections.innerHTML = list
      .filter(s => s.groups && Object.keys(s.groups).length)
      .map(s => renderSection(s.title, s.note, s.groups))
      .join("");
  } catch (err) {
    emptyBlock.textContent = "Couldn't reach the analysis service. Check that the server is running.";
    emptyBlock.hidden = false;
  }
})();
