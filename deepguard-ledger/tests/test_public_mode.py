"""The public link's rules and the tunnel launcher, without the models or the
internet: the rate limits, the visitor cookie, cloudflared's output, and a
stand-in cloudflared. The rules wired into the real app are tested at the
end of test_ledger_app.py."""
import sys
import textwrap
from pathlib import Path

import pytest

LEDGER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEDGER_DIR))
import public_mode as pm  # noqa: E402
import tunnel  # noqa: E402

COOKIE = "0123456789abcdef0123456789abcdef"


# ------------------------------------------------------------- rate limits

def test_a_sliding_window_per_key():
    limiter = pm.RateLimiter()
    limit = [(("browser", "1.2.3.4", COOKIE, "heavy"), 2, 60)]
    assert limiter.admit(limit, now=0) and limiter.admit(limit, now=10)
    assert not limiter.admit(limit, now=59)
    assert limiter.admit(limit, now=61)  # the first one has left the window


def test_a_refused_request_uses_up_nothing():
    limiter = pm.RateLimiter()
    roomy, full = (("a",), 10, 60), (("b",), 1, 60)
    assert limiter.admit([roomy, full], now=0)
    for t in range(1, 6):
        assert not limiter.admit([roomy, full], now=t)  # "b" is full...
    # ...and "a" was not charged for the five refusals: 9 more still fit.
    assert all(limiter.admit([roomy], now=10) for _ in range(9))
    assert not limiter.admit([roomy], now=10)


def test_a_classroom_on_one_wifi_shares_a_bigger_allowance(monkeypatch):
    monkeypatch.setattr(pm, "RATES", {"heavy": (2, 60), "write": (1, 60), "any": (100, 60)})
    monkeypatch.setattr(pm, "NETWORK_RATES", {"heavy": (3, 60), "write": (5, 60), "any": (100, 60)})
    limiter = pm.RateLimiter()
    phone_a, phone_b = "a" * 32, "b" * 32
    assert limiter.admit(pm.limits_for("10.0.0.1", phone_a, "heavy"), now=0)
    assert limiter.admit(pm.limits_for("10.0.0.1", phone_a, "heavy"), now=1)
    assert not limiter.admit(pm.limits_for("10.0.0.1", phone_a, "heavy"), now=2)  # phone A's share
    assert limiter.admit(pm.limits_for("10.0.0.1", phone_b, "heavy"), now=3)      # phone B still gets one
    assert not limiter.admit(pm.limits_for("10.0.0.1", phone_b, "heavy"), now=4)  # the Wi-Fi's share is used up
    assert limiter.admit(pm.limits_for("10.0.0.2", phone_b, "heavy"), now=5)      # another network is separate


def test_a_registration_also_counts_as_a_heavy_check():
    limits = pm.limits_for("10.0.0.1", None, "write")
    assert {key[-1] for key, _, _ in limits} == {"write", "heavy"}
    assert all(key[0] == "network" or key[2] == "-" for key, _, _ in limits)  # no cookie: one shared key


def test_idle_keys_are_forgotten(monkeypatch):
    monkeypatch.setattr(pm.RateLimiter, "MAX_KEYS", 10)
    limiter = pm.RateLimiter()
    for i in range(10):
        limiter.admit([((i,), 5, 60)], now=0)
    limiter.admit([(("new",), 5, 60)], now=1000)
    assert list(limiter._seen) == [("new",)]


def test_buckets():
    assert pm.bucket_of("POST", "/api/ledger/register") == "write"
    assert pm.bucket_of("POST", "/api/analyze") == "heavy"
    assert pm.bucket_of("GET", "/api/analyze") == "any"
    assert pm.bucket_of("GET", "/explorer") == "any"
    assert "/api/registry/register" in pm.BLOCKED_PATHS


# ------------------------------------------------------- visitors' cookies

def test_only_our_own_cookie_values_count():
    assert pm.browser_of({pm.COOKIE: COOKIE}) == COOKIE
    for bad in ("", "short", COOKIE.upper(), COOKIE + "0", "x" * 32, "../../etc"):
        assert pm.browser_of({pm.COOKIE: bad}) is None
    assert pm.browser_of({}) is None
    assert pm.browser_of({pm.COOKIE: pm.new_browser_id()}) is not None


def test_a_browser_tag_is_a_short_hash_not_the_cookie():
    tag = pm.browser_tag(COOKIE)
    assert len(tag) == 16 and tag not in COOKIE and tag == pm.browser_tag(COOKIE)
    assert pm.browser_tag(None) is None


def test_public_rules_are_off_unless_the_server_is_public(monkeypatch):
    headers = {"cf-connecting-ip": "1.2.3.4"}
    monkeypatch.setattr(pm, "ENABLED", False)
    assert not pm.is_public_request(headers)
    monkeypatch.setattr(pm, "ENABLED", True)
    assert pm.is_public_request(headers) and pm.is_public_request({"cf-ray": "8f1-BOM"})
    assert not pm.is_public_request({"host": "127.0.0.1:8000"})


# --------------------------------------------------------------- the tunnel

BOX = [
    "2026-11-05T09:12:01Z INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, is a "
    "quick way to experiment and try it out.",
    "2026-11-05T09:12:01Z INF Requesting new quick Tunnel on trycloudflare.com...",
    "2026-11-05T09:12:04Z INF +--------------------------------------------------------------------------------------------+",
    "2026-11-05T09:12:04Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |",
    "2026-11-05T09:12:04Z INF |  https://seasonal-deck-organisms-sf.trycloudflare.com                                      |",
]


def test_reading_the_address_cloudflared_prints():
    assert [tunnel.parse_url(line) for line in BOX] == [None] * 4 + ["https://seasonal-deck-organisms-sf.trycloudflare.com"]
    failure = ('2026-11-05T09:12:31Z ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": '
               'dial tcp: lookup api.trycloudflare.com: no such host')
    assert tunnel.parse_url(failure) is None
    assert tunnel.parse_url("see https://a-b.trycloudflare.com.example.com") is None  # the host must end there
    assert tunnel.parse_url("|  https://a-b.trycloudflare.com  |") == "https://a-b.trycloudflare.com"
    assert tunnel.parse_url(None) is None


def test_finding_cloudflared(tmp_path, monkeypatch):
    monkeypatch.setattr(tunnel, "HERE", tmp_path)
    monkeypatch.setattr(tunnel, "WINDOWS_INSTALL_DIRS", ())
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: None)
    monkeypatch.delenv("DEEPGUARD_CLOUDFLARED", raising=False)
    assert tunnel.find_cloudflared() is None
    (tmp_path / "tools").mkdir()
    in_tools = tmp_path / "tools" / tunnel.EXE
    in_tools.write_bytes(b"")
    assert tunnel.find_cloudflared() == str(in_tools)
    chosen = tmp_path / "elsewhere.exe"
    chosen.write_bytes(b"")
    monkeypatch.setenv("DEEPGUARD_CLOUDFLARED", str(chosen))
    assert tunnel.find_cloudflared() == str(chosen)
    assert tunnel.find_cloudflared(str(in_tools)) == str(in_tools)  # --cloudflared wins
    assert tunnel.find_cloudflared(str(tmp_path / "missing.exe")) == str(chosen)


def _stand_in(tmp_path: Path, body: str) -> list:
    script = tmp_path / "fake_cloudflared.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, "-I", str(script)]


def test_a_tunnel_reports_its_address_and_stops(tmp_path):
    fake = _stand_in(tmp_path, f"""
        import sys, time
        assert sys.argv[1:4] == ["tunnel", "--no-autoupdate", "--url"], sys.argv
        for line in {BOX!r}:
            print(line, file=sys.stderr, flush=True)
        for i in range(600):                      # keeps talking, like the real one
            print("INF Registered tunnel connection", i, file=sys.stderr, flush=True)
            time.sleep(0.05)
    """)
    t = tunnel.Tunnel(fake, 8000, "Deep-Guard")
    try:
        assert t.wait_for_url(timeout=20) == "https://seasonal-deck-organisms-sf.trycloudflare.com"
        assert t.process.poll() is None
    finally:
        t.stop()
    assert t.process.poll() is not None


def test_a_tunnel_that_fails_says_why(tmp_path):
    fake = _stand_in(tmp_path, """
        import sys
        print("INF Requesting new quick Tunnel on trycloudflare.com...", file=sys.stderr)
        print('ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": '
              'dial tcp: i/o timeout', file=sys.stderr)
        sys.exit(1)
    """)
    t = tunnel.Tunnel(fake, 8000, "Deep-Guard")
    assert t.wait_for_url(timeout=20) is None
    assert "i/o timeout" in t.failure()
    t.stop()


def test_public_without_cloudflared_explains_how_to_get_it(monkeypatch, capsys):
    import run_deepguard
    monkeypatch.setattr(tunnel, "find_cloudflared", lambda explicit=None: None)
    monkeypatch.setattr(sys, "argv", ["run_deepguard.py", "--public"])
    started = []
    monkeypatch.setattr(run_deepguard.subprocess, "Popen", lambda *a, **k: started.append(a))
    with pytest.raises(SystemExit) as exit_info:
        run_deepguard.main()
    assert exit_info.value.code == 1 and not started  # nothing was started
    out = capsys.readouterr().out
    assert "winget install --id Cloudflare.cloudflared" in out and "cloudflared-windows-amd64.exe" in out
