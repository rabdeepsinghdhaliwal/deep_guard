// Deep-Guard — Bias Map page behaviour. Plain JS, no build step, same as script.js.

// Defense in depth, same reasoning as registry.js's escapeHtml: `label`
// below comes from labeled_test_set.csv (generator/subject/style values),
// which today needs filesystem access to edit -- lower risk than
// registry.js's remotely-writable title, but "currently harder to reach"
// isn't a reason to render it unescaped if that CSV is ever fed by a
// less-trusted source later (e.g. a crowd-sourced labeling flow).
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function accuracyClass(entry) {
  if (!entry.trustworthy) return "is-unsure";
  if (entry.accuracy >= 0.8) return "is-good";
  if (entry.accuracy >= 0.5) return "is-mid";
  return "is-bad";
}

function renderGrid(container, groups) {
  const entries = Object.entries(groups).sort((a, b) => a[1].accuracy - b[1].accuracy);
  container.innerHTML = entries.map(([label, entry]) => `
    <div class="bias-map__cell ${accuracyClass(entry)}">
      <div class="bias-map__cell-label">${escapeHtml(label)}</div>
      <div class="bias-map__cell-value">${Math.round(entry.accuracy * 100)}%</div>
      <div class="bias-map__cell-n">n=${entry.sample_count}${entry.trustworthy ? "" : " (too small to trust)"}</div>
    </div>
  `).join("");
}

(async function load() {
  const scopeWarning     = document.getElementById("scopeWarning");
  const overallBlock     = document.getElementById("overallBlock");
  const overallAccuracy  = document.getElementById("overallAccuracy");
  const totalImages      = document.getElementById("totalImages");
  const generatedAt      = document.getElementById("generatedAt");
  const bySubjectSection = document.getElementById("bySubjectSection");
  const byStyleSection   = document.getElementById("byStyleSection");
  const byGeneratorSection = document.getElementById("byGeneratorSection");
  const byComboSection   = document.getElementById("byComboSection");
  const emptyBlock       = document.getElementById("emptyBlock");

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

    overallAccuracy.textContent = Math.round(data.overall_accuracy * 100) + "%";
    totalImages.textContent = data.total_images;
    generatedAt.textContent = new Date(data.generated_at_utc).toLocaleDateString();
    overallBlock.hidden = false;

    renderGrid(document.getElementById("bySubjectGrid"), data.by_subject);
    bySubjectSection.hidden = false;

    renderGrid(document.getElementById("byStyleGrid"), data.by_style);
    byStyleSection.hidden = false;

    renderGrid(document.getElementById("byGeneratorGrid"), data.by_generator);
    byGeneratorSection.hidden = false;

    renderGrid(document.getElementById("byComboGrid"), data.by_combination);
    byComboSection.hidden = false;
  } catch (err) {
    emptyBlock.textContent = "Couldn't reach the analysis service. Check that the server is running.";
    emptyBlock.hidden = false;
  }
})();
