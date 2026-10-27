"""
Start Deep-Guard with the ledger: two witness nodes, then the main server.

    cd deepguard-ledger
    ../deepguard-bouncer/venv/Scripts/python.exe run_deepguard.py

Then open http://127.0.0.1:8000 (Analysis), /ledger and /explorer.
Ctrl+C stops all three.

    --no-witnesses   start only the main server (the ledger still works;
                     blocks just get no co-signatures until witnesses run)
    --port 8000      the main server's port
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WITNESSES = (("Witness A", 8101), ("Witness B", 8102))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-Guard + ledger + witnesses")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-witnesses", action="store_true")
    args = parser.parse_args()

    children = []
    if not args.no_witnesses:
        for name, port in WITNESSES:
            env = {**os.environ, "DEEPGUARD_WITNESS_NAME": name}
            children.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "witness:app", "--host", "127.0.0.1", "--port", str(port),
                 "--log-level", "warning"], cwd=str(HERE), env=env))
            print(f"  started {name} on http://127.0.0.1:{port}")
        time.sleep(1.0)

    print(f"\n  Deep-Guard is starting on http://127.0.0.1:{args.port}  (models take ~20 s to load)")
    print(f"    Analysis  http://127.0.0.1:{args.port}/")
    print(f"    Ledger    http://127.0.0.1:{args.port}/ledger")
    print(f"    Explorer  http://127.0.0.1:{args.port}/explorer")
    print("  Press Ctrl+C to stop everything.\n")

    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    try:
        import uvicorn
        uvicorn.run("ledger_app:app", host="127.0.0.1", port=args.port)
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
        print("  Stopped Deep-Guard and its witnesses.")


if __name__ == "__main__":
    main()
