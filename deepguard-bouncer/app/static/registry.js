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

const explainSection = document.getElementById("explainSection");

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
// The same applies to everything the explanations below render: names,
// positions, colours and image data all come from the server, so text is
// escaped, colours must look like #rrggbb, numbers go through Number(),
// and image data must be plain base64 before it reaches an attribute.
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const num = (value, digits = 1) => Number(value).toFixed(digits);
const pct = (fraction) => Math.round(Number(fraction) * 100);
const isHex = (value) => typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value);
const safeBase64 = (value) => (typeof value === "string" && /^[A-Za-z0-9+/=]+$/.test(value) ? value : "");

// One colour per shared "thing", reused by its list row, its points and
// its lines in the figure — the lists double as the figure's legend.
const THING_COLOURS = ["#3A4CB4", "#2E7D5B", "#B07A1E", "#B23B34", "#2B7A8C", "#7A4E9C", "#6F7A1F", "#C2567A", "#55504A"];
const thingColour = (index) => THING_COLOURS[Math.abs(Number(index)) % THING_COLOURS.length];

function joinWords(words) {
  const safe = words.map(escapeHtml);
  return safe.length <= 1 ? safe.join("") : `${safe.slice(0, -1).join(", ")} and ${safe[safe.length - 1]}`;
}

function withArticle(noun) {
  if (noun === "digital art") return noun;
  return (/^[aeiou]/i.test(noun) ? "an " : "a ") + noun;
}

let figureCounter = 0;

// An area's name, as SVG text at (x, y): split onto two lines when it
// wouldn't fit a third of a narrow figure ("dark background").
function svgLabel(text, x, y, attrs = "") {
  const words = String(text).split(" ");
  if (text.length <= 12 || words.length < 2) {
    return `<text x="${num(x)}" y="${num(y)}" ${attrs}>${escapeHtml(text)}</text>`;
  }
  const cut = Math.ceil(words.length / 2);
  return `<text x="${num(x)}" y="${num(y)}" ${attrs}>`
    + `<tspan x="${num(x)}">${escapeHtml(words.slice(0, cut).join(" "))}</tspan>`
    + `<tspan x="${num(x)}" dy="1.15em">${escapeHtml(words.slice(cut).join(" "))}</tspan></text>`;
}

// ------------------------------------------------------ distinct features --

function featuresFigure(df) {
  const [w, h] = df.size.map(Number);
  // Drawn at roughly the size it's shown (half the page column), so the
  // labels and points come out at a readable, consistent size.
  const W = 320;
  const H = Math.max(80, Math.round((W * h) / w));
  const grid = [1, 2].map((k) =>
    `<line x1="${num((W * k) / 3)}" y1="0" x2="${num((W * k) / 3)}" y2="${H}"/>` +
    `<line x1="0" y1="${num((H * k) / 3)}" x2="${W}" y2="${num((H * k) / 3)}"/>`
  ).join("");
  const points = df.points.drawn.map(([x, y]) =>
    `<circle cx="${num(x * W)}" cy="${num(y * H)}" r="1.9"/>`
  ).join("");
  const labels = df.areas.map((area, i) =>
    svgLabel(area.name || area.position, ((i % 3) * W) / 3 + 5, (Math.floor(i / 3) * H) / 3 + 14)
  ).join("");
  return `
    <svg class="features__svg" viewBox="0 0 ${W} ${H}" role="img"
         aria-label="The image with its strongest distinctive points and the name of each of its nine areas">
      <image href="data:image/jpeg;base64,${safeBase64(df.preview_jpeg_base64)}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none"/>
      <g class="features__grid">${grid}</g>
      <g class="features__points">${points}</g>
      <g class="features__labels">${labels}</g>
    </svg>`;
}

function renderDistinctFeatures(df, title) {
  if (!df) return "";
  const named = Boolean(df.namer);
  const heading = title
    ? `What the registry will recognise &ldquo;${escapeHtml(title)}&rdquo; by`
    : "What the registry sees in your upload";
  const count = Number(df.points.count);
  const drawn = Math.min(df.points.drawn.length, count);
  const things = df.things.map((t) => `
      <li class="features__thing">
        <span class="features__thing-name">${escapeHtml(t.name)}</span>
        <span class="features__thing-where">${joinWords(t.positions)}</span>
        <span class="features__thing-points">${Number(t.points).toLocaleString()} points</span>
      </li>`).join("");
  const swatches = df.palette.filter((c) => isHex(c.hex)).map((c) => `
      <span class="swatch" style="--swatch: ${c.hex}; flex-grow: ${Math.max(Number(c.share), 0.04)}" title="${c.hex} · ${pct(c.share)}%">
        <span class="swatch__share">${pct(c.share)}%</span>
      </span>`).join("");
  const medium = df.medium
    ? `<section><h4 class="features__h">Overall</h4><p>Looks like ${escapeHtml(withArticle(df.medium.name))}.</p></section>`
    : "";
  const stored = df.stored_kb
    ? `<p class="features__stored">Kept by the registry: the points' positions and descriptions, the area names and an
       8&times;8 grid of average colours (${Number(df.stored_kb)} KB) &mdash; not the picture.</p>`
    : "";
  return `
    <article class="features" id="${title ? "features" : "upload-features"}">
      <p class="features__kicker">Distinct features</p>
      <h3 class="features__title">${heading}</h3>
      <div class="features__layout">
        <figure class="features__figure">
          ${featuresFigure(df)}
          <figcaption>The ${drawn.toLocaleString()} strongest of ${count.toLocaleString()} distinctive points, and
            ${named ? "what each ninth of the picture looks like" : "the nine areas the picture is divided into"}.</figcaption>
        </figure>
        <div class="features__facts">
          <section>
            <h4 class="features__h">Distinctive points</h4>
            <p>${count.toLocaleString()} corners and blobs (SIFT) whose descriptions stay recognisable after cropping,
               rotating, resizing and recolouring.</p>
          </section>
          <section>
            <h4 class="features__h">${named ? "Things in it" : "Areas"}</h4>
            <ul class="features__things">${things}</ul>
            <p class="features__note">${named
              ? `Named by ${escapeHtml(df.namer)} &mdash; a model's guess, so read each as &ldquo;looks like&rdquo;.`
              : "No namer is installed on this server, so areas are named by position."}</p>
          </section>
          <section>
            <h4 class="features__h">Colour signature</h4>
            <div class="swatches">${swatches}</div>
          </section>
          ${medium}
          ${stored}
        </div>
      </div>
    </article>`;
}

// --------------------------------------------------------- why it matched --

function differences(ex) {
  const g = ex.geometry;
  const items = [];
  items.push(g.shown_position === "almost all of it"
    ? "Shows almost all of the original"
    : `Shows <strong>${pct(g.original_area_shown)}%</strong> of the original, from the ${escapeHtml(g.shown_position || "middle")}`);
  if (Number(g.added_area) >= 0.05) {
    items.push(`${pct(g.added_area)}% of the upload isn't from the original (a border, background or padding added around it)`);
  }
  const rot = Number(g.rotation_deg);
  items.push(rot === 0 ? "Not rotated" : `Rotated <strong>${num(Math.abs(rot))}&deg;</strong> ${rot > 0 ? "anticlockwise" : "clockwise"}`);
  items.push(g.mirrored ? "<strong>Mirrored</strong> (flipped left to right)" : "Not mirrored");
  const size = Number(g.size_ratio);
  items.push(size >= 0.95 && size <= 1.05 ? "Same size as the original"
    : size > 1 ? `Enlarged <strong>${num(size)}&times;</strong>` : `Shrunk to <strong>${pct(size)}%</strong> of the original's size`);
  const colour = {
    removed: "<strong>Colour removed</strong> (black and white)",
    recoloured: "<strong>Colours changed</strong> (tinted or recoloured)",
    faded: "<strong>Colours weakened</strong>",
    intensified: "<strong>Colours intensified</strong>",
    kept: "Colours unchanged",
  }[ex.colour.change];
  if (colour) items.push(colour);
  if (ex.colour.brightness === "brighter") items.push("<strong>Made brighter</strong>");
  if (ex.colour.brightness === "darker") items.push("<strong>Made darker</strong>");
  return items.map((item) => `<li>${item}</li>`).join("");
}

function shareRows(parts, similarity, withPoints) {
  return parts.map((part) => {
    const ofMatch = Number(part.share) / Number(similarity);
    const width = Math.max(0, Math.min(100, ofMatch * 100));
    const colour = part.index !== undefined ? thingColour(part.index) : "#8A837A";
    return `
      <li class="why__thing">
        <span class="why__chip" style="--chip: ${colour}"></span>
        <span class="why__thing-name">${escapeHtml(part.name)}<span class="why__thing-where">${joinWords(part.positions)}</span></span>
        <span class="why__thing-bar"><span style="width: ${num(width)}%; background: ${colour}"></span></span>
        <span class="why__thing-share">${ofMatch > 0 ? pct(ofMatch) + "%" : "&minus;"}</span>
        ${withPoints ? `<span class="why__thing-points">${Number(part.shared_points).toLocaleString()} matching points</span>` : ""}
      </li>`;
  }).join("");
}

function matchFigure(ex, uf) {
  const f = ex.figure;
  const id = ++figureCounter;
  const H = 420;
  const [uw, uh] = uf.size.map(Number);
  const Wl = Math.round((H * uw) / uh);
  const Wr = Math.round(H * Number(f.original_aspect));
  const gap = 40;
  const ox = Wl + gap;
  const W = ox + Wr;

  const cw = Wr / 8;
  const ch = H / 8;
  const wash = f.colour_grid.flatMap((row, r) => row.map((hex, c) => isHex(hex)
    ? `<rect x="${num(ox + c * cw)}" y="${num(r * ch)}" width="${num(cw + 1)}" height="${num(ch + 1)}" fill="${hex}"/>` : "")).join("");
  const stored = f.original_points.map(([x, y]) => `<circle cx="${num(ox + x * Wr)}" cy="${num(y * H)}" r="1.6"/>`).join("");

  const sharedThings = new Set(ex.common.map((t) => t.index));
  const areaGrid = [1, 2].map((k) =>
    `<line x1="${num(ox + (Wr * k) / 3)}" y1="0" x2="${num(ox + (Wr * k) / 3)}" y2="${H}"/>` +
    `<line x1="${ox}" y1="${num((H * k) / 3)}" x2="${W}" y2="${num((H * k) / 3)}"/>`).join("");
  const areaNames = f.area_names.map((name, i) =>
    svgLabel(name, ox + ((i % 3) * Wr) / 3 + 6, (Math.floor(i / 3) * H) / 3 + 16,
      `class="${sharedThings.has(f.area_things[i]) ? "is-shared" : "is-missing"}"`)
  ).join("");
  const footprint = f.footprint.map(([x, y]) => `${num(ox + x * Wr)},${num(y * H)}`).join(" ");

  const links = f.lines.map(([qx, qy, rx, ry, t]) =>
    `<line x1="${num(qx * Wl)}" y1="${num(qy * H)}" x2="${num(ox + rx * Wr)}" y2="${num(ry * H)}" stroke="${thingColour(t)}"/>`).join("");
  const ends = f.lines.map(([qx, qy, rx, ry, t]) =>
    `<circle cx="${num(qx * Wl)}" cy="${num(qy * H)}" r="3.2" fill="${thingColour(t)}"/>` +
    `<circle cx="${num(ox + rx * Wr)}" cy="${num(ry * H)}" r="3.2" fill="${thingColour(t)}"/>`).join("");

  return `
    <figure class="why__figure">
      <svg class="why__svg" viewBox="0 0 ${W} ${H}" role="img"
           aria-label="Your upload on the left, the registered original drawn from its stored points on the right, with lines joining the distinctive points they share">
        <defs>
          <clipPath id="whyClip${id}"><rect x="${ox}" y="0" width="${Wr}" height="${H}"/></clipPath>
          <filter id="whyBlur${id}" x="0" y="0" width="100%" height="100%">
            <feGaussianBlur stdDeviation="${num(Math.min(cw, ch) * 0.45)}"/>
          </filter>
        </defs>
        <image href="data:image/jpeg;base64,${safeBase64(uf.preview_jpeg_base64)}" x="0" y="0" width="${Wl}" height="${H}" preserveAspectRatio="none"/>
        <g clip-path="url(#whyClip${id})">
          <rect class="why__paper" x="${ox}" y="0" width="${Wr}" height="${H}"/>
          <g class="why__wash" filter="url(#whyBlur${id})">${wash}</g>
          <g class="why__stored">${stored}</g>
          <g class="why__areas">${areaGrid}${areaNames}</g>
          <polygon class="why__footprint" points="${footprint}"/>
        </g>
        <rect class="why__frame" x="${ox}" y="0.5" width="${Wr - 0.5}" height="${H - 1}"/>
        <g class="why__links">${links}</g>
        <g class="why__ends">${ends}</g>
      </svg>
      <div class="why__figure-labels" style="grid-template-columns: ${Wl}fr ${gap}fr ${Wr}fr">
        <span>Your upload</span><span></span>
        <span>The registered original, drawn from what the registry kept: its distinctive points and an 8&times;8 colour grid
          (it keeps no pixels). The dashed outline is where your upload sits inside it.</span>
      </div>
    </figure>`;
}

function renderExplanation(match, uf, i) {
  const ex = match.explanation || {};
  const similarity = Number(ex.similarity ?? match.similarity);
  let body;
  if (!ex.available) {
    body = `<p class="why__unavailable">${escapeHtml(ex.reason || "No explanation is available for this match.")}</p>`;
  } else if (!ex.confirmed) {
    const n = Number(ex.shared_points);
    body = `
      <p class="why__verdict is-unconfirmed"><strong>${pct(similarity)}% content similarity</strong>, but only
        ${n} distinctive point${n === 1 ? "" : "s"} agree on where the copy would sit in the original &mdash; too few to
        confirm it point by point (unrelated images shared up to 9 by chance in our tests). The registry's network still
        judged the content similar; treat this match with more caution.</p>
      <h4 class="why__h">Parts of your upload the network matched</h4>
      <ul class="why__things">${shareRows(ex.upload_parts, similarity, false)}</ul>`;
  } else {
    const outside = Number(ex.not_from_original_share);
    const outsideNote = Math.abs(outside) >= 0.005
      ? `<p class="why__note">${outside < 0
          ? `Content that isn't from the original pulled the similarity down by ${Math.round(-outside * 100)} percentage point${Math.round(-outside * 100) === 1 ? "" : "s"}.`
          : `Content outside the original added ${Math.round(outside * 100)} percentage points.`}</p>`
      : "";
    const missing = ex.only_in_original.length
      ? `<ul class="why__missing">${ex.only_in_original.map((t) =>
          `<li><span class="why__chip is-hollow" style="--chip: ${thingColour(t.index)}"></span>${escapeHtml(t.name)}
           <span class="why__thing-where">${joinWords(t.positions)}</span></li>`).join("")}</ul>`
      : `<p class="why__note">Nothing &mdash; every part of the original appears in the upload, at least partly.</p>`;
    body = `
      <p class="why__verdict"><strong>${pct(similarity)}% content similarity</strong>, ${ex.strength === "strong" ? "strongly " : ""}confirmed
        by <strong>${Number(ex.shared_points).toLocaleString()}</strong> distinctive points that the two images share and that
        all agree on where the copy sits in the original.</p>
      <div class="why__columns">
        <section>
          <h4 class="why__h">Common to both &mdash; and how much each counted</h4>
          <ul class="why__things">${shareRows(ex.common, similarity, true)}</ul>
          ${outsideNote}
          <p class="why__note">The percentage is each part's share of the registry's own score. Matching points are
            distinctive points found in that part of both images &mdash; smooth or dark areas have few, so a part can
            still count without any.</p>
        </section>
        <section>
          <h4 class="why__h">How this copy differs</h4>
          <ul class="why__diffs">${differences(ex)}</ul>
          <h4 class="why__h">Only in the original</h4>
          ${missing}
        </section>
      </div>
      ${uf ? matchFigure(ex, uf) : ""}`;
  }
  return `
    <article class="why" id="why-${i}">
      <p class="why__kicker">Why it matched</p>
      <h3 class="why__title">&ldquo;${escapeHtml(match.title)}&rdquo;</h3>
      ${body}
      <p class="why__method">How this is worked out: the similarity is the registry's decision (Meta's SSCD network).
        Its split between the parts comes from that same network &mdash; Shapley values over its grid of feature cells &mdash;
        and adds up to the full percentage. Matching points are SIFT distinctive points that agree on one placement
        (RANSAC). ${ex.namer ? "Names are CLIP's guess of what each area looks like." : ""}</p>
    </article>`;
}

function showExplanations(html) {
  explainSection.innerHTML = html;
  explainSection.hidden = !html;
}

// ------------------------------------------------------------ handlers ----

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
    const df = record.distinct_features;
    showResult(registerResult, `
      <p><strong>&#10003; Registered as &ldquo;${escapeHtml(record.title)}&rdquo;</strong></p>
      <dl class="registry__fields">
        <div><dt>Exact fingerprint</dt><dd>${escapeHtml(record.sha256_merkle_root.slice(0, 16))}&hellip;</dd></div>
        <div><dt>Signed at</dt><dd>${escapeHtml(new Date(record.timestamp_utc).toLocaleString())}</dd></div>
        ${df ? `<div><dt>Distinctive points</dt><dd>${Number(df.points.count).toLocaleString()}</dd></div>` : ""}
      </dl>
      ${df ? `<a class="registry__why-link" href="#features">See its distinct features &darr;</a>` : ""}
    `);
    showExplanations(renderDistinctFeatures(df, record.title));
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
    const uf = data.upload_features;

    if (data.matches.length === 0) {
      showResult(checkResult, `<p>No match found against the ${escapeHtml(data.registry_size)} registered item(s) — this doesn't appear to share content with anything registered.</p>`);
      showExplanations(uf ? `<details class="features__details" open><summary>Distinct features of your upload</summary>${renderDistinctFeatures(uf, null)}</details>` : "");
    } else {
      showResult(checkResult, `
        <p><strong>${data.matches.length} match(es) found:</strong></p>
        ${data.matches.map((m, i) => `
          <div class="registry__match">
            <div class="registry__match-title">${escapeHtml(m.title)}</div>
            <div class="registry__match-sim">${Math.round(m.similarity * 100)}% content similarity</div>
            <div class="registry__match-meta">Registered ${escapeHtml(new Date(m.timestamp_utc).toLocaleDateString())} &middot; source: ${escapeHtml(m.source)}</div>
            <a class="registry__why-link" href="#why-${i}">Why it matched &darr;</a>
          </div>
        `).join("")}
      `);
      showExplanations(
        data.matches.map((m, i) => renderExplanation(m, uf, i)).join("")
        + (uf ? `<details class="features__details"><summary>Distinct features of your upload</summary>${renderDistinctFeatures(uf, null)}</details>` : "")
      );
    }
  } catch (err) {
    showResult(checkResult, escapeHtml(err.message) || "Something went wrong checking the registry.", true);
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check registry";
  }
});
