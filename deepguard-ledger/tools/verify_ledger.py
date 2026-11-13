"""
Check the Deep-Guard ledger -- or one receipt -- with no server running.

This is what an outsider would run: it needs only Python, the
`cryptography` package, ledger_core.py and Deep-Guard's PUBLIC key from a
source they trust. It never uses the key written inside the blocks or the
receipt to check themselves -- a forger could put their own key there.

    python tools/verify_ledger.py chain                      (defaults below)
    python tools/verify_ledger.py chain ledger_data/chain.jsonl ../deepguard-bouncer/models/demo_signer_public.pem
    python tools/verify_ledger.py receipt deepguard-receipt-1234.json ../deepguard-bouncer/models/demo_signer_public.pem
    python tools/verify_ledger.py anchors                    (every Bitcoin anchor, looked up online)
    python tools/verify_ledger.py anchors --offline          (everything except the Bitcoin lookup)

Exit code 0 = everything checks out, 1 = something does not, 2 = could not check.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ledger_core  # noqa: E402

DEFAULT_CHAIN = HERE.parent / "ledger_data" / "chain.jsonl"
DEFAULT_KEY = HERE.parent.parent / "deepguard-bouncer" / "models" / "demo_signer_public.pem"


def _print_checks(checks: list[dict], indent: str = "    ") -> None:
    for c in checks:
        mark = "ok  " if c["ok"] is True else "FAIL" if c["ok"] is False else "...."  # None: not decided yet
        print(f"{indent}[{mark}] {c['name']}" + (f" -- {c['detail']}" if c.get("detail") and c["ok"] is not True else ""))


def _read_chain(chain_path: Path) -> list[dict]:
    blocks = []
    for line_no, line in enumerate(chain_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            blocks.append(json.loads(line))
        except json.JSONDecodeError:
            blocks.append({"_unreadable": line_no, "raw": line[:300]})
    return blocks


def check_chain(chain_path: Path, key_path: Path) -> int:
    public_pem = key_path.read_bytes()
    blocks = _read_chain(chain_path)
    report = ledger_core.verify_chain(blocks, public_pem)
    print(f"Deep-Guard ledger: {chain_path}")
    print(f"Checked against public key {ledger_core.key_id(public_pem)} from {key_path}\n")
    for block in report["blocks"]:
        status = "ok  " if block["ok"] else "FAIL"
        print(f"  block #{block['index']:<3} [{status}] {str(block.get('block_hash'))[:16]}…")
        if not block["ok"]:
            _print_checks([c for c in block["checks"] if not c["ok"]])
    print()
    if report["ok"]:
        print(f"RESULT: intact -- {report['block_count']} blocks, {report['record_count']} records, every hash, "
              f"Merkle root, link and signature checks out.")
        return 0
    print(f"RESULT: TAMPERED -- first bad block is #{report['first_bad_index']} "
          f"(bad blocks: {', '.join('#' + str(i) for i in report['bad_indexes'])}).")
    return 1


def check_receipt(receipt_path: Path, key_path: Path) -> int:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result = ledger_core.verify_receipt(receipt, key_path.read_bytes())
    record = receipt["record"]
    print(f"Receipt for '{record.get('title') or record.get('type')}' (record {record.get('record_id')}) "
          f"in block #{receipt['block_header']['index']}\n")
    _print_checks(result["checks"], indent="  ")
    print()
    print("RESULT: genuine -- this record is in that signed block." if result["ok"] else "RESULT: does NOT check out.")
    return 0 if result["ok"] else 1


def check_anchors(data_dir: Path, offline: bool) -> int:
    """Each anchor: its header file still hashes to the anchored block hash,
    the chain on disk still holds that block, and the proof reaches the
    Merkle root of the Bitcoin block it names (looked up on two independent
    block explorers, unless --offline)."""
    import anchoring  # needs the opentimestamps package; chain and receipt checks do not

    index_path = data_dir / "anchors" / "anchors.json"
    if not index_path.exists():
        print(f"No anchors in {data_dir} yet. Anchor a block from the explorer page first.")
        return 2
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    chain_path = data_dir / "chain.jsonl"
    blocks = _read_chain(chain_path) if chain_path.exists() else []
    print(f"Deep-Guard Bitcoin anchors: {index_path}")
    print("Bitcoin blocks looked up on " + ("nothing (--offline)" if offline else
                                           " / ".join(name for name, _ in anchoring.EXPLORERS)) + "\n")
    all_ok, confirmed = True, 0
    for entry in sorted(entries, key=lambda e: e["block_index"]):
        i = entry["block_index"]
        header = (data_dir / "anchors" / entry["header_file"]).read_bytes()
        proof = (data_dir / "anchors" / entry["proof_file"]).read_bytes()
        chain_hash = blocks[i].get("block_hash") if 0 <= i < len(blocks) else None
        result = anchoring.verify_files(header, proof, anchored_hash=entry["block_hash"], chain_hash=chain_hash,
                                        lookup=None if offline else anchoring.bitcoin_block)
        where = ""
        if result["bitcoin"]:
            confirmed += 1
            where = f" -> Bitcoin block #{result['bitcoin']['height']}, mined {result['bitcoin']['time_utc']}"
        mark = "ok  " if result["ok"] else "FAIL"
        print(f"  block #{i:<3} [{mark}] {result['status']}{where}")
        _print_checks([c for c in result["checks"] if c["ok"] is not True], indent="      ")
        all_ok = all_ok and result["ok"]
    print()
    if not all_ok:
        print("RESULT: an anchor does NOT check out -- see the failed lines above.")
        return 1
    print(f"RESULT: every anchor checks out -- {confirmed} of {len(entries)} confirmed in Bitcoin"
          + (" (offline: Bitcoin not looked up)." if offline else "."))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("chain", "receipt", "anchors"):
        print(__doc__)
        return 2
    try:
        if argv[1] == "anchors":
            rest = [a for a in argv[2:] if a != "--offline"]
            data_dir = Path(rest[0]) if rest else DEFAULT_CHAIN.parent
            return check_anchors(data_dir, offline="--offline" in argv[2:])
        if argv[1] == "chain":
            chain = Path(argv[2]) if len(argv) > 2 else DEFAULT_CHAIN
            key = Path(argv[3]) if len(argv) > 3 else DEFAULT_KEY
            return check_chain(chain, key)
        if len(argv) < 3:
            print("Give the receipt file: python tools/verify_ledger.py receipt <receipt.json> [public key]")
            return 2
        key = Path(argv[3]) if len(argv) > 3 else DEFAULT_KEY
        return check_receipt(Path(argv[2]), key)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not check: {exc}")
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv))
