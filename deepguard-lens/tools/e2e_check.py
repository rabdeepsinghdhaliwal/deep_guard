"""
Deep-Guard Lens, end to end: load this unpacked extension into Microsoft
Edge (headless), open a gallery page on a made-up website showing the
pictures you name, run the same check the right-click menu runs on each,
print every badge and save a screenshot. Read-only: it checks, it never
registers anything.

    (start Deep-Guard first: python deepguard-ledger/run_deepguard.py)
    python deepguard-lens/tools/e2e_check.py test_images/real3.jpg path/to/other.jpg ...
    python deepguard-lens/tools/e2e_check.py --server https://xyz.trycloudflare.com picture.jpg

Needs Playwright (pip install playwright) and Edge, which Windows 11 has.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

EXT = Path(__file__).resolve().parents[1]
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pictures", nargs="+", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--screenshot", type=Path, default=Path("lens_gallery.png"))
    args = parser.parse_args()
    files = {f"{i}-{p.name}": p for i, p in enumerate(args.pictures, start=1)}
    page_html = ("<!doctype html><title>Gallery</title><style>body{font:15px system-ui;margin:24px}"
                 "figure{display:inline-block;margin:0 18px 18px 0}img{width:300px;display:block}</style>"
                 "<h1>Pictures shared this week</h1>" + "".join(
                     f'<figure><img src="https://gallery.example/{n}"><figcaption>{p.name}</figcaption></figure>'
                     for n, p in files.items()))

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            tempfile.mkdtemp(prefix="lens-"), channel="msedge", headless=True, viewport={"width": 1400, "height": 1000},
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])

        def serve(route):
            name = route.request.url.rsplit("/", 1)[-1]
            if name in files:
                route.fulfill(body=files[name].read_bytes(), content_type=TYPES.get(files[name].suffix.lower(), "image/jpeg"))
            else:
                route.fulfill(body=page_html, content_type="text/html")
        ctx.route("https://gallery.example/**", serve)
        worker = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=20000)
        worker.evaluate("async (s) => chrome.storage.local.set({server: s})", args.server)
        page = ctx.new_page()
        page.goto("https://gallery.example/index.html")
        page.wait_for_function("[...document.images].every(i => i.complete)", timeout=20000)
        tab = worker.evaluate("async () => (await chrome.tabs.query({url: 'https://gallery.example/*'}))[0].id")
        failures = 0
        for name, path in files.items():
            result = worker.evaluate("async ([t, u]) => await self.dgLens.checkImage(t, u)", [tab, f"https://gallery.example/{name}"])
            if result.get("error"):
                failures += 1
                print(f"{path.name}: could not check: {result['error']}")
            else:
                match = (result.get("match") or {}).get("title")
                print(f"{path.name}: {result['verdict']} · {result['headline']}" + (f" · “{match}”" if match else "")
                      + f" · Content Credentials: {result['credentials']['verdict']}")
        page.wait_for_timeout(400)
        page.screenshot(path=str(args.screenshot), full_page=True)
        print(f"screenshot: {args.screenshot}")
        ctx.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
