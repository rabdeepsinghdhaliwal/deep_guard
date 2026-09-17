// Deep-Guard — Content Registry page behaviour. Plain JS, no build step.

const registerForm   = document.getElementById("registerForm");
const registerFile   = document.getElementById("registerFile");
const registerTitle  = document.getElementById("registerTitle");
const registerBtn    = document.getElementById("registerBtn");
const registerResult = document.getElementById("registerResult");

const checkForm   = document.getElementById("checkForm");
const checkFile   = document.getElementById("checkFile");
const checkBtn    = document.getElementById("checkBtn");
const checkResult = document.getElementById("checkResult");

function showResult(el, html, isError = false) {
  el.innerHTML = html;
  el.classList.toggle("is-error", isError);
  el.hidden = false;
}

// SECURITY FIX (found during the round-1 adversarial review): title is
// free text a user types in, stored on the server, and later rendered
// back into ANOTHER visitor's page when their check matches it. Without
// this, registering something titled e.g. `<img src=x onerror=...>`
// would execute arbitrary script in the browser of anyone who later
// searches and matches it -- confirmed live: the server stored and
// returned that exact payload verbatim, with no server-side sanitising
// either. Every piece of server-supplied text below is escaped before
// going into innerHTML, not just title -- escaping only the one field
// tested is how the next untrusted field quietly reopens the same hole.
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

registerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!registerFile.files[0]) return;

  registerBtn.disabled = true;
  registerBtn.textContent = "Registering…";

  const body = new FormData();
  body.append("file", registerFile.files[0]);
  body.append("title", registerTitle.value);

  try {
    const res = await fetch("/api/registry/register", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned an error (${res.status}).`);
    }
    const record = await res.json();
    showResult(registerResult, `
      <p><strong>&#10003; Registered as &ldquo;${escapeHtml(record.title)}&rdquo;</strong></p>
      <dl class="registry__fields">
        <div><dt>Exact fingerprint</dt><dd>${escapeHtml(record.sha256_merkle_root.slice(0, 16))}&hellip;</dd></div>
        <div><dt>Signed at</dt><dd>${escapeHtml(new Date(record.timestamp_utc).toLocaleString())}</dd></div>
      </dl>
    `);
    registerForm.reset();
  } catch (err) {
    showResult(registerResult, escapeHtml(err.message) || "Something went wrong registering that image.", true);
  } finally {
    registerBtn.disabled = false;
    registerBtn.textContent = "Register";
  }
});

checkForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!checkFile.files[0]) return;

  checkBtn.disabled = true;
  checkBtn.textContent = "Checking…";

  const body = new FormData();
  body.append("file", checkFile.files[0]);

  try {
    const res = await fetch("/api/registry/check", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned an error (${res.status}).`);
    }
    const data = await res.json();

    if (data.matches.length === 0) {
      showResult(checkResult, `<p>No match found against the ${escapeHtml(data.registry_size)} registered item(s) — this doesn't appear to share content with anything registered.</p>`);
    } else {
      showResult(checkResult, `
        <p><strong>${data.matches.length} match(es) found:</strong></p>
        ${data.matches.map(m => `
          <div class="registry__match">
            <div class="registry__match-title">${escapeHtml(m.title)}</div>
            <div class="registry__match-sim">${Math.round(m.similarity * 100)}% content similarity</div>
            <div class="registry__match-meta">Registered ${escapeHtml(new Date(m.timestamp_utc).toLocaleDateString())} &middot; source: ${escapeHtml(m.source)}</div>
          </div>
        `).join("")}
      `);
    }
  } catch (err) {
    showResult(checkResult, escapeHtml(err.message) || "Something went wrong checking the registry.", true);
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check registry";
  }
});
