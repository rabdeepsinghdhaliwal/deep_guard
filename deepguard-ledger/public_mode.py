"""
Deep-Guard Stage 4 — the live public link: what changes for visitors.

`python run_deepguard.py --public` puts the laptop's server on the internet
through a Cloudflare quick tunnel (a free, temporary https address, no
account). Anyone with the link, or the QR code on the projector, can then use
Deep-Guard from their own phone. That needs a few rules, and they apply only
to requests that came through the tunnel (Cloudflare marks every one with a
CF-Connecting-IP header). The presenter's own browser at 127.0.0.1 keeps
every feature.

For visitors:
  * Uploads are capped at 15 MB (the laptop's 200 MB limit is for local videos).
  * Each phone gets a fair share: a few heavy checks (analysing, verifying,
    registering) per minute. A whole classroom on one Wi-Fi shares a single
    internet address, so phones are told apart by a cookie, and the address
    as a whole gets a bigger share of its own.
  * Heavy checks run one at a time, the presenter's included. Grad-CAM hangs
    hooks on the shared detector model while it works, so two checks running
    side by side could swap heatmaps between people.
  * Actions that belong to the owner are off: anchoring to Bitcoin (it calls
    out to public calendars), revoking, disputing and resolving (creator keys
    are held by the server, so on a public link anyone could type someone
    else's name), and filling the older content registry.
  * Creator names already used on this laptop are reserved, and a name a
    visitor makes stays with the browser that made it.
  * Local file paths are left out of answers, and every page says it is a
    public demo and what happens to a picture that is checked.
"""

import hashlib
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Iterable, Optional

ENABLED = os.environ.get("DEEPGUARD_PUBLIC") == "1"
PUBLIC_URL = os.environ.get("DEEPGUARD_PUBLIC_URL", "").rstrip("/")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
HEAVY_AT_ONCE = 1
HEAVY_WAIT_SECONDS = 40  # Cloudflare gives up on an answer after 100 s

# (requests, per seconds). RATES is for one browser; NETWORK_RATES for
# everyone behind one internet address (a classroom on the college Wi-Fi).
RATES = {"heavy": (12, 60), "write": (6, 60), "any": (300, 60)}
NETWORK_RATES = {"heavy": (60, 60), "write": (30, 60), "any": (3000, 60)}

HEAVY_PATHS = {"/api/analyze", "/api/explain/shap-regions", "/api/fingerprint/compression-test",
               "/api/registry/check", "/api/registry/register", "/api/ledger/verify", "/api/ledger/register",
               "/api/credentials/read"}
WRITE_PATHS = {"/api/ledger/register", "/api/ledger/seal"}
BLOCKED_PATHS = {
    "/api/ledger/anchor": "Anchoring to Bitcoin is done by the owner, on the laptop itself.",
    "/api/ledger/anchors/upgrade": "Checking Bitcoin confirmations is done by the owner, on the laptop itself.",
    "/api/ledger/revoke": "Corrections are off on the public link: creator keys are held by this server, so anyone "
                          "could type someone else's name. Use the laptop.",
    "/api/ledger/dispute": "Corrections are off on the public link. Use the laptop.",
    "/api/ledger/resolve": "Resolving disputes is the owner's job, on the laptop itself.",
    "/api/registry/register": "The content registry is filled from the laptop. To register a picture of your own, "
                              "use the Ledger page: it records you as its creator.",
}

BANNER = ('<div class="dgl-public-banner" role="note"><b>Public demo</b> on a student project laptop. Pictures you '
          'check are never stored; registering one writes only its fingerprints to the ledger. Corrections and '
          'Bitcoin anchoring are off here.</div>')

COOKIE = "dg_visitor"
COOKIE_MAX_AGE = 24 * 60 * 60
_COOKIE_VALUE = re.compile(r"^[0-9a-f]{32}$")


def is_public_request(headers) -> bool:
    """True for a request that came in through the Cloudflare tunnel. The
    tunnel connects from this machine too, but Cloudflare adds these headers
    to every request it forwards, and a visitor cannot remove them."""
    return ENABLED and ("cf-connecting-ip" in headers or "cf-ray" in headers)


def network_of(headers, client_host: Optional[str]) -> str:
    """The visitor's internet address, as Cloudflare saw it."""
    return headers.get("cf-connecting-ip") or client_host or "unknown"


def browser_of(cookies) -> Optional[str]:
    """This browser's visitor cookie, or None if it has none (yet) or it
    isn't one of ours."""
    value = cookies.get(COOKIE)
    return value if value and _COOKIE_VALUE.match(value) else None


def new_browser_id() -> str:
    return secrets.token_hex(16)


def browser_tag(browser: Optional[str]) -> Optional[str]:
    """What a visitor's creator identity remembers about the browser that
    made it. A hash, because identities are public (/api/ledger/creators) and
    the cookie itself works like a key."""
    return hashlib.sha256(browser.encode("ascii")).hexdigest()[:16] if browser else None


def bucket_of(method: str, path: str) -> str:
    if method == "POST" and path in WRITE_PATHS:
        return "write"
    if method == "POST" and path in HEAVY_PATHS:
        return "heavy"
    return "any"


def limits_for(network: str, browser: Optional[str], bucket: str) -> list[tuple]:
    """Every (key, limit, window) a request counts against. A registration
    is also a heavy check. Requests without a cookie share one key per
    network."""
    buckets = ("write", "heavy") if bucket == "write" else (bucket,)
    out = []
    for b in buckets:
        out.append((("network", network, b), *NETWORK_RATES[b]))
        out.append((("browser", network, browser or "-", b), *RATES[b]))
    return out


class RateLimiter:
    """A sliding window per key: the times of recent requests, forgotten
    once older than the window. A request is let in only if every limit it
    counts against has room, and then it is counted against all of them."""

    MAX_KEYS = 20000

    def __init__(self):
        self._seen = defaultdict(deque)
        self._lock = threading.Lock()

    def admit(self, limits: Iterable[tuple], now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        limits = list(limits)
        with self._lock:
            for key, limit, window in limits:
                times = self._seen[key]
                while times and now - times[0] > window:
                    times.popleft()
                if len(times) >= limit:
                    return False
            for key, _, _ in limits:
                self._seen[key].append(now)
            if len(self._seen) > self.MAX_KEYS:
                self._forget_idle(now)
            return True

    def _forget_idle(self, now: float) -> None:
        longest = max(window for _, window in (*RATES.values(), *NETWORK_RATES.values()))
        for key in [k for k, times in self._seen.items() if not times or now - times[-1] > longest]:
            del self._seen[key]
