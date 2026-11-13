# Deep-Guard Lens

A browser extension for Microsoft Edge and Google Chrome. **Right-click any
picture on any website → "Check with Deep-Guard"**, and a badge is pinned on
the picture with what your Deep-Guard server found:

- the ledger verdict (1 authentic & unaltered · 2 same picture, file modified ·
  3 no provenance · 4 declared AI-generated), the record's title, creator and
  block, and how it matched (exact file, hidden ID read from the pixels, looks
  the same, same content)
- the picture's Content Credentials (the CR badge): none, signed by a signer on
  a C2PA trust list, signed by this Deep-Guard server, signer not proven,
  expired, or changed after signing
- for a picture with no record, the AI detector's reading, labelled as a guess

The toolbar button opens a small panel: which Deep-Guard server to ask (your
own laptop, or the public link from `run_deepguard.py --public`), whether it is
answering, a **Check the images on this page** button, and the last checks.

## Install (once, about a minute)

1. Open `edge://extensions` (Chrome: `chrome://extensions`).
2. Switch on **Developer mode** (bottom left in Edge, top right in Chrome).
3. Click **Load unpacked** and choose this `deepguard-lens` folder.
4. Optional: pin it with the puzzle-piece button so the shield sits in the toolbar.

Then start Deep-Guard (`python run_deepguard.py` in `deepguard-ledger`). The
extension asks `http://127.0.0.1:8000` unless you set another address in its panel.

## How it works

`background.js` (the service worker) adds the right-click item, downloads the
picture itself (the extension may read any site's images, which is why it asks
for access to all sites), and sends it to the server's `/api/ledger/verify`.
`content.js` is only injected into a page when a check starts; it draws the
badge inside a shadow root, so the website and the badge can't restyle each
other. Everything a badge says comes from the server's answer; the extension
decides nothing on its own, and nothing is stored except the last 12 checks
in the browser's session storage (gone when the browser closes).

## Check it end to end

`tools/e2e_check.py` loads the extension into a headless Edge, opens a
made-up gallery page showing the pictures you name, runs the right-click check
on each one, prints the badges and saves a screenshot (read-only):

```
python deepguard-lens/tools/e2e_check.py test_images/real3.jpg generalization_test/held_out_images/dalle3/dalle3_010.webp
```

## Limits

- Not published to the Edge or Chrome stores: it is loaded as an unpacked
  developer extension.
- Pictures drawn on a `<canvas>` or set as CSS backgrounds have no right-click
  "image" item; only `<img>` pictures can be checked.
- The server must be reachable from the browser: your own laptop, or the public
  link when checking from another machine.
