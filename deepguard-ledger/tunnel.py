"""
Deep-Guard Stage 4 — the tunnels behind `run_deepguard.py --public`.

In plain words: a Cloudflare quick tunnel gives a program on this laptop a
temporary public https address, like https://four-random-words.trycloudflare.com,
with no account and no router settings. `cloudflared tunnel --url
http://127.0.0.1:8000` connects OUT to Cloudflare, and Cloudflare sends
visitors' requests back down that connection, so it works on college Wi-Fi
that would never let anyone connect in. The address is new on every start
and dies when cloudflared stops. Cloudflare offers these for testing and
development, with no uptime guarantee and at most 200 requests in flight
per tunnel: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/

Every request that arrives this way carries Cloudflare's CF-Connecting-IP
header, which is how public_mode.py tells visitors from the presenter.
"""

import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
EXE = "cloudflared.exe" if os.name == "nt" else "cloudflared"
# Where `winget install --id Cloudflare.cloudflared` puts it: PATH only
# catches up in a terminal opened after the install.
WINDOWS_INSTALL_DIRS = (r"C:\Program Files (x86)\cloudflared", r"C:\Program Files\cloudflared")
START_TIMEOUT = 60

# A quick tunnel's name is a few random words joined by dashes. Requiring a
# dash keeps out api.trycloudflare.com, which turns up in cloudflared's error
# messages when it can't reach Cloudflare.
URL = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com(?![a-z0-9.-])")

INSTALL_HELP = f"""
  --public needs cloudflared, Cloudflare's free tunnel program (a single
  file, about 55 MB, no account needed). Either:

    winget install --id Cloudflare.cloudflared        (then open a NEW terminal)

  or download  cloudflared-windows-amd64.exe  from
    https://github.com/cloudflare/cloudflared/releases/latest
  and save it as
    {HERE / "tools" / EXE}

  You can also point to it with --cloudflared PATH or the DEEPGUARD_CLOUDFLARED variable.
"""


def find_cloudflared(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [explicit, os.environ.get("DEEPGUARD_CLOUDFLARED"), shutil.which("cloudflared"),
                  str(HERE / "tools" / EXE)]
    if os.name == "nt":
        candidates += [str(Path(d) / EXE) for d in WINDOWS_INSTALL_DIRS]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def parse_url(line: str) -> Optional[str]:
    """cloudflared prints the address it was given in a box on stderr:
    '... INF |  https://some-random-words.trycloudflare.com  |'."""
    found = URL.search(line or "")
    return found.group(0) if found else None


class Tunnel:
    """One `cloudflared tunnel --url` process for one local port.
    `cloudflared` is its path (or a command list, which the tests use to run
    a stand-in)."""

    def __init__(self, cloudflared, port: int, name: str):
        self.name, self.port, self.url = name, port, None
        self.log = deque(maxlen=40)
        self._settled = threading.Event()
        command = [cloudflared] if isinstance(cloudflared, str) else list(cloudflared)
        self.process = subprocess.Popen(
            [*command, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self._read, name=f"cloudflared-{port}", daemon=True).start()

    def _read(self) -> None:
        # Keeps reading after the address shows up: a full pipe would stall
        # cloudflared. The last lines are kept for error messages.
        for line in self.process.stderr:
            self.log.append(line.rstrip())
            if self.url is None and parse_url(line):
                self.url = parse_url(line)
                self._settled.set()
        self._settled.set()  # cloudflared exited

    def wait_for_url(self, timeout: float = START_TIMEOUT) -> Optional[str]:
        self._settled.wait(timeout)
        return self.url

    def failure(self) -> str:
        tail = [line for line in self.log if " ERR " in line or "error" in line.lower()] or list(self.log)
        return "\n".join("      " + line for line in tail[-6:]) or "      (cloudflared printed nothing)"

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def answers(url: str, timeout: float = 10) -> bool:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Deep-Guard launcher"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 — any failure just means "not yet"
        return False


def wait_until_answers(url: str, within: float, every: float = 3) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if answers(url):
            return True
        time.sleep(every)
    return False
