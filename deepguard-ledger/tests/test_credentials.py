"""Content Credentials (C2PA): our own certificate authority, signing a
credentialed copy, and reading credentials Deep-Guard did and did not make.
No models and no network: just the c2pa SDK and three official test files."""
import io
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID
from PIL import Image

LEDGER_DIR = Path(__file__).resolve().parents[1]
REPO = LEDGER_DIR.parent
sys.path.insert(0, str(LEDGER_DIR))
sys.path.insert(0, str(REPO / "deepguard-bouncer" / "app"))  # evidence.py
import credentials  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "c2pa"
BITS = "1" + "0110" * 15


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return credentials.CredentialStore(tmp_path_factory.mktemp("creds"))


@pytest.fixture(scope="module")
def picture():
    original = (REPO / "test_images" / "real3.jpg").read_bytes()
    buffer = io.BytesIO()
    Image.open(io.BytesIO(original)).convert("RGB").save(buffer, "JPEG", quality=95)
    return original, buffer.getvalue()


def sign(store, picture, origin="camera"):
    original, copy = picture
    return store.sign(copy, "image/jpeg", title="real3.credentialed.jpg", record_id="rec-123", creator="Rabdeep",
                      declared_origin=origin, watermark_alg="com.adobe.trustmark.Q", watermark_schema_code=1,
                      watermark_bits=BITS, ledger_key_id="0123456789abcdef", parent=(original, "image/jpeg", "real3.jpg"))


# ------------------------------------------------------------ our own CA

def test_the_signing_certificate_follows_the_c2pa_rules(store):
    signer = x509.load_pem_x509_certificate(store.signer_pem)
    ca = x509.load_pem_x509_certificate(store.ca_pem)
    assert signer.issuer == ca.subject
    basic = signer.extensions.get_extension_for_class(x509.BasicConstraints)
    assert basic.critical and not basic.value.ca
    usage = signer.extensions.get_extension_for_class(x509.KeyUsage)
    assert usage.critical and usage.value.digital_signature and not usage.value.key_cert_sign
    eku = signer.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.EMAIL_PROTECTION in eku and credentials.DOCUMENT_SIGNING in eku
    assert ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE not in eku
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca


def test_the_identity_is_made_once_and_kept(store):
    again = credentials.CredentialStore(store.dir)
    assert again.ca_pem == store.ca_pem and again.ca_fingerprint == store.ca_fingerprint


# ------------------------------------------------------------ writing and reading our own

def test_a_signed_copy_reads_as_ours_and_names_its_watermark_and_record(store, picture):
    summary = store.read(sign(store, picture))
    assert summary["verdict"] == "ours" and summary["issues"] == [], summary
    assert summary["soft_binding"] == {"alg": "com.adobe.trustmark.Q", "value": "1*" + BITS}
    assert summary["deepguard"]["record_id"] == "rec-123" and summary["deepguard"]["creator"] == "Rabdeep"
    assert [a["action"] for a in summary["actions"]] == ["c2pa.opened", "c2pa.watermarked"]
    assert summary["ingredients"] == 1 and not summary["ai_declared"]


def test_an_ai_declared_copy_carries_the_standard_ai_label(store, picture):
    summary = store.read(sign(store, picture, "ai-generated"))
    assert summary["verdict"] == "ours" and summary["ai_declared"]
    assert summary["digital_source_type"] == "trainedAlgorithmicMedia"
    assert summary["actions"][0]["action"] == "c2pa.created"
    assert credentials.to_evidence(summary)["points_to"] == "ai"


def test_one_changed_byte_reads_as_altered(store, picture):
    signed = bytearray(sign(store, picture))
    signed[-400] ^= 0xFF
    summary = store.read(bytes(signed))
    assert summary["verdict"] == "altered"
    assert any(i["code"] == "assertion.dataHash.mismatch" for i in summary["issues"])


def test_a_re_save_strips_the_credentials(store, picture):
    """Why the watermark exists: metadata does not survive a re-save."""
    buffer = io.BytesIO()
    Image.open(io.BytesIO(sign(store, picture))).save(buffer, "JPEG", quality=85)
    assert store.read(buffer.getvalue())["verdict"] == "none"


def test_another_installation_cannot_vouch_for_our_signer(store, picture, tmp_path):
    other = credentials.CredentialStore(tmp_path / "other")
    summary = other.read(sign(store, picture))
    assert summary["verdict"] == "unknown"  # signed and unchanged, but nobody it trusts signed it


# ------------------------------------------------------------ credentials we did not make

@pytest.mark.parametrize("name, verdict", [
    ("adobe-20220124-A.jpg", "none"),
    ("adobe-20220124-C.jpg", "unknown"),
    ("adobe-20220124-E-dat-CA.jpg", "altered"),
])
def test_official_c2pa_test_files(store, name, verdict):
    summary = store.read((FIXTURES / name).read_bytes())
    assert summary["verdict"] == verdict, summary
    if verdict != "none":
        assert summary["signer"] == "C2PA Test Signing Cert"


def test_garbage_is_reported_not_crashed_on(store):
    assert store.read(b"not a picture at all")["verdict"] in ("none", "error")


@pytest.mark.parametrize("verdict", ["none", "trusted", "ours", "unknown", "expired", "altered", "problem", "error"])
def test_every_verdict_fits_the_evidence_format(verdict):
    summary = {"present": verdict != "none", "verdict": verdict, "headline": credentials.VERDICT_TEXT[verdict],
               "ai_declared": False, "capture_declared": True}
    item = credentials.to_evidence(summary)
    assert item["layer"] == 0 and item["layer_name"] == "Content Credentials"
    assert (item["strength"] == "proof") == (verdict == "trusted")
