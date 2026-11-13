"""
Start Deep-Guard with the ledger: two witness nodes, then the main server.

    cd deepguard-ledger
    ../deepguard-bouncer/venv/Scripts/python.exe run_deepguard.py

Then open http://127.0.0.1:8000 (Analysis), /ledger and /explorer.
Ctrl+C stops everything.

    --no-witnesses   start only the main server (the ledger still works;
                     blocks just get no co-signatures until witnesses run)
    --port 8000      the main server's port
    --public         also put the demo on the internet, so anyone can try it
                     from their phone: a free Cloudflare quick tunnel (no
                     account) for the server and one for each witness. Prints
                     the link and a QR code; put http://127.0.0.1:8000/share
                     on the projector. Needs cloudflared (see tunnel.py). The
                     limits visitors get are in public_mode.py; this laptop
                     keeps every feature.
    --cloudflared P  where cloudflared is, if it isn't on PATH or in tools/
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WITNESSES = (("Witness A", 8101), ("Witness B", 8102))


def _stop(processes) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _open_tunnels(tunnel, cloudflared: str, port: int, witnesses) -> tuple[list, dict]:
    """Starts every tunnel at once, then waits for their addresses. Returns
    (tunnels, environment for the server). Exits if the main one fails."""
    print("  Opening the public link (Cloudflare quick tunnels, ~10 s)...")
    main = tunnel.Tunnel(cloudflared, port, "Deep-Guard")
    side = [tunnel.Tunnel(cloudflared, witness_port, name) for name, witness_port in witnesses]
    opened = [main, *side]
    if main.wait_for_url() is None:
        print("\n  The public link could not be opened. cloudflared said:\n" + main.failure())
        print("\n  Is this laptop online? Some networks block Cloudflare's tunnel ports (7844);"
              " a phone hotspot usually works.\n")
        for t in opened:
            t.stop()
        raise SystemExit(1)
    witness_urls = {}
    for t in side:
        if t.wait_for_url() is None:
            print(f"  ! {t.name} got no public address, so visitors' explorers will show it as not answering:\n"
                  + t.failure())
        else:
            witness_urls[t.name] = t.url
    env = {"DEEPGUARD_PUBLIC": "1", "DEEPGUARD_PUBLIC_URL": main.url,
           "DEEPGUARD_WITNESS_PUBLIC_URLS": ",".join(f"{name}={url}" for name, url in witness_urls.items())}
    return opened, env


def _print_qr(url: str) -> None:
    """A QR code in the terminal, when there is a terminal (the /share page
    has a sharper one for the projector)."""
    if not sys.stdout.isatty():
        return
    try:
        import segno
        segno.make(url, error="m").terminal(border=2)
    except Exception:  # noqa: BLE001 — the QR is a convenience; the link is printed anyway
        pass


def _announce_when_live(tunnel, port: int, url: str, witness_urls: str) -> None:
    """Waits for the server to finish loading, then for the public link to
    answer from outside, and says so (or says what to check)."""
    if not tunnel.wait_until_answers(f"http://127.0.0.1:{port}/api/health", within=240, every=2):
        return
    if tunnel.wait_until_answers(f"{url}/api/public/status", within=120):
        lines = [f"  The public link is live:   {url}",
                 f"  Projector page (QR code):  http://127.0.0.1:{port}/share"]
        for part in filter(None, witness_urls.split(",")):
            name, witness_url = part.split("=", 1)
            if not tunnel.answers(f"{witness_url}/witness/info"):
                lines.append(f"  ! {name} doesn't answer over its public address yet.")
        bar = "  " + "=" * 64
        print("\n".join(["", bar, *lines, bar, ""]), flush=True)
    else:
        print(f"\n  ! The public link {url} hasn't answered for 2 minutes. Check that this laptop is online; "
              "some networks block Cloudflare tunnels (a phone hotspot usually works).\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-Guard + ledger + witnesses")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-witnesses", action="store_true")
    parser.add_argument("--public", action="store_true", help="also open a public link for phones (cloudflared)")
    parser.add_argument("--cloudflared", help="path to cloudflared, if it isn't on PATH or in tools/")
    args = parser.parse_args()
    try:  # the link should show up at once even when the output goes to a log file
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    sys.path.insert(0, str(HERE))
    import tunnel

    cloudflared = None
    if args.public:
        cloudflared = tunnel.find_cloudflared(args.cloudflared)
        if cloudflared is None:
            print(tunnel.INSTALL_HELP)
            raise SystemExit(1)

    children, tunnels = [], []
    try:
        witnesses = () if args.no_witnesses else WITNESSES
        for name, port in witnesses:
            env = {**os.environ, "DEEPGUARD_WITNESS_NAME": name, **({"DEEPGUARD_PUBLIC": "1"} if args.public else {})}
            children.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "witness:app", "--host", "127.0.0.1", "--port", str(port),
                 "--log-level", "warning"], cwd=str(HERE), env=env))
            print(f"  started {name} on http://127.0.0.1:{port}")
        if witnesses:
            time.sleep(1.0)

        if args.public:
            tunnels, public_env = _open_tunnels(tunnel, cloudflared, args.port, witnesses)
            # Read by public_mode.py when the server imports it, just below.
            os.environ.update(public_env)
            url = public_env["DEEPGUARD_PUBLIC_URL"]
            print(f"\n  Public link:  {url}\n  (waiting for Deep-Guard to load before it answers)\n")
            _print_qr(url)
            threading.Thread(target=_announce_when_live, name="public-link-check", daemon=True,
                             args=(tunnel, args.port, url, public_env["DEEPGUARD_WITNESS_PUBLIC_URLS"])).start()

        print(f"\n  Deep-Guard is starting on http://127.0.0.1:{args.port}  (models take ~20 s to load)")
        print(f"    Analysis  http://127.0.0.1:{args.port}/")
        print(f"    Ledger    http://127.0.0.1:{args.port}/ledger")
        print(f"    Explorer  http://127.0.0.1:{args.port}/explorer")
        if args.public:
            print(f"    Share     http://127.0.0.1:{args.port}/share   <- the QR code, for the projector")
        print("  Press Ctrl+C to stop everything.\n")

        os.chdir(HERE)
        import uvicorn
        uvicorn.run("ledger_app:app", host="127.0.0.1", port=args.port)
    finally:
        for t in tunnels:
            t.stop()
        _stop(children)
        if tunnels:
            print("  Closed the public link.")
        print("  Stopped Deep-Guard" + (" and its witnesses." if children else "."))


if __name__ == "__main__":
    main()
