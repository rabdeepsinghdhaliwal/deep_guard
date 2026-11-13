"""
Deep-Guard Stage 4 — Content Credentials (C2PA): reading anyone's, writing our own.

In plain words: C2PA is the industry standard behind the little "CR" mark
(Adobe, Google, OpenAI, Nikon, Leica, LinkedIn, TikTok...). A file can carry
a signed "manifest" inside it saying who made it, with what, what was done to
it, and whether AI made it. Reading one answers "who signed this, and has it
changed since?" without trusting the file's sender. Writing one lets
Deep-Guard's credentialed copies speak that same language to every other
C2PA reader.

READING: any upload is checked. The answer is one of five, in plain words:
  none          the file carries no Content Credentials (most files today)
  trusted       signed by a signer on a public trust list (the official C2PA
                Trust List, or the interim list contentcredentials.org used),
                and nothing changed since
  ours          signed by THIS Deep-Guard server's own demo certificate
                authority, and nothing changed since
  unknown       signed, and nothing changed since, but no trust list knows the
                signer: who signed it is not proven
  altered       the file or its credentials changed after signing
  (and "expired" / "problem" for a signer whose certificate is no longer
  valid, without the content having changed)

WRITING: the credentialed copy made at registration (watermark.py) is signed
with a manifest that says, in C2PA's own vocabulary: Deep-Guard watermarked
it (`c2pa.watermarked`), the hidden ID it carries (`c2pa.soft-binding`,
algorithm `com.adobe.trustmark.Q`, exactly as TrustMark's own C2PA example
writes it), and which ledger record it belongs to (`com.deepguard.ledger`).
A file declared AI-generated says so with the IPTC digital source type
`trainedAlgorithmicMedia`, which other C2PA readers display as "AI-generated".
This is the C2PA's "durable Content Credentials" design: manifest +
watermark + fingerprint, because a re-save strips the manifest and the
watermark is what survives.

Honest limit: signing needs an X.509 certificate. Deep-Guard creates its own
small certificate authority, once per installation (credentials/, never
committed). Our pages trust it and say so; outside validators such as
contentcredentials.org/verify will show the manifest but call the signer
unknown. Certificates on the official list need the C2PA conformance programme.
"""

import datetime as dt
import io
import json
import threading
from pathlib import Path
from typing import Optional

import c2pa
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

HERE = Path(__file__).resolve().parent
TRUST_DIR = HERE / "trust"
PUBLIC_TRUST_LISTS = (TRUST_DIR / "C2PA-TRUST-LIST.pem", TRUST_DIR / "interim-trust-list-anchors.pem")
DEFAULT_CREDENTIALS_DIR = HERE / "credentials"

ORGANISATION = "Deep-Guard (student project)"
CA_COMMON_NAME = "Deep-Guard Demo Root CA"
SIGNER_COMMON_NAME = "Deep-Guard Credentials Signer"
DOCUMENT_SIGNING = ObjectIdentifier("1.3.6.1.5.5.7.3.36")  # id-kp-documentSigning, one of the two EKUs C2PA allows
LEDGER_ASSERTION = "com.deepguard.ledger"

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
SOURCE_TYPES = {
    "trainedAlgorithmicMedia": "made by generative AI",
    "compositeWithTrainedAlgorithmicMedia": "edited or combined with generative AI",
    "algorithmicMedia": "made by an algorithm",
    "compositeSynthetic": "a composite including synthetic parts",
    "algorithmicallyEnhanced": "enhanced by an algorithm",
    "digitalCapture": "captured with a digital camera",
    "computationalCapture": "captured by a camera with computational processing",
    "compositeCapture": "a composite of camera captures",
    "negativeFilm": "scanned from negative film",
    "positiveFilm": "scanned from positive film",
    "print": "scanned from a print",
    "screenCapture": "a screenshot",
    "digitalCreation": "made digitally by a person",
    "humanEdits": "edited by a person",
    "virtualRecording": "recorded from a virtual environment",
    "dataDrivenMedia": "made from data",
}
AI_SOURCE_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia",
                   "compositeSynthetic"}
CAPTURE_SOURCE_TYPES = {"digitalCapture", "computationalCapture", "compositeCapture", "negativeFilm", "positiveFilm"}

ISSUE_TEXT = {
    "signingCredential.untrusted": "the signer's certificate is not on a trust list this server knows",
    "signingCredential.expired": "the signer's certificate had expired, and no trusted time-stamp shows it was "
                                 "valid when it signed",
    "signingCredential.revoked": "the signer's certificate was revoked",
    "signingCredential.invalid": "the signer's certificate is not valid",
    "claimSignature.mismatch": "the signature does not match the credentials: they were altered after signing",
    "assertion.dataHash.mismatch": "the file's bytes changed after it was signed",
    "assertion.hashedURI.mismatch": "part of the credentials was swapped after signing",
    "assertion.bmffHash.mismatch": "the file's bytes changed after it was signed",
    "assertion.boxesHash.mismatch": "the file's bytes changed after it was signed",
}

# How a person reads each answer (and how strongly it should weigh, in the
# evidence format of deepguard-bouncer/app/evidence.py: layer 0).
VERDICT_TEXT = {
    "none": "This file carries no Content Credentials. Most files don't yet; it proves nothing either way.",
    "trusted": "Signed by {signer}, a signer on a public C2PA trust list, and nothing has changed since it was signed.",
    "ours": "Signed by this Deep-Guard server's own demo certificate authority, and nothing has changed since.",
    "unknown": "Signed by “{signer}” and nothing has changed since, but no trust list this server knows vouches "
               "for that signer, so who signed it is not proven.",
    "expired": "Signed by {signer}, but the signing certificate is no longer valid. The content itself was not "
               "changed.",
    "altered": "Content Credentials are present, but this file or its credentials changed after they were signed.",
    "problem": "Content Credentials are present but could not be checked: {issue}.",
    "error": "The Content Credentials in this file could not be read.",
}


def _short_source_type(uri: str) -> str:
    return (uri or "").rstrip("/").split("/")[-1]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Our own small certificate authority (one per installation)
# ---------------------------------------------------------------------------

def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANISATION)])


def make_identity() -> dict:
    """A root CA and the one signing certificate it issues, both Ed25519. The
    signing certificate follows the C2PA rules for claim signers: not a CA,
    Key Usage digitalSignature (critical), an Extended Key Usage of email
    protection and document signing, never anyExtendedKeyUsage."""
    now = _now()
    ca_key = Ed25519PrivateKey.generate()
    ca = (x509.CertificateBuilder().subject_name(_name(CA_COMMON_NAME)).issuer_name(_name(CA_COMMON_NAME))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=3650))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                       data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                       crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, None))
    signer_key = Ed25519PrivateKey.generate()
    signer = (x509.CertificateBuilder().subject_name(_name(SIGNER_COMMON_NAME)).issuer_name(_name(CA_COMMON_NAME))
              .public_key(signer_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=730))
              .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
              .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                           data_encipherment=False, key_agreement=False, key_cert_sign=False,
                                           crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION, DOCUMENT_SIGNING]),
                             critical=False)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(signer_key.public_key()), critical=False)
              .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
              .sign(ca_key, None))
    pem = lambda c: c.public_bytes(serialization.Encoding.PEM)  # noqa: E731
    key_pem = lambda k: k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,  # noqa: E731
                                        serialization.NoEncryption())
    return {"ca.pem": pem(ca), "ca.key.pem": key_pem(ca_key), "signer.pem": pem(signer),
            "signer.key.pem": key_pem(signer_key)}


class CredentialStore:
    """Signs credentialed copies and reads anyone's Content Credentials.

    credentials/ holds this installation's CA and signing key (made on first
    use; the private keys never leave the folder, which git ignores)."""

    def __init__(self, credentials_dir: Path = DEFAULT_CREDENTIALS_DIR, trust_lists=PUBLIC_TRUST_LISTS):
        self.dir = Path(credentials_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        files = {name: self.dir / name for name in ("ca.pem", "ca.key.pem", "signer.pem", "signer.key.pem")}
        if not all(p.exists() for p in files.values()):
            for name, data in make_identity().items():
                files[name].write_bytes(data)
        self.ca_pem = files["ca.pem"].read_bytes()
        self.signer_pem = files["signer.pem"].read_bytes()
        self._signer_key_pem = files["signer.key.pem"].read_bytes()
        self.ca_fingerprint = x509.load_pem_x509_certificate(self.ca_pem).fingerprint(hashes.SHA256()).hex()
        self.public_anchors = "\n".join(Path(p).read_text(encoding="ascii") for p in trust_lists if Path(p).exists())
        self._lock = threading.Lock()

    # ------------------------------------------------------------- writing
    def sign(self, data: bytes, mime: str, *, title: str, record_id: str, creator: str, declared_origin: str,
             watermark_alg: str, watermark_schema_code: int, watermark_bits: str, ledger_key_id: str,
             parent: Optional[tuple[bytes, str, str]] = None) -> bytes:
        """Returns `data` with a signed manifest embedded. parent = (bytes,
        mime, file name) of the original the copy was made from; it becomes
        the manifest's parent ingredient. An AI-declared file is a "created"
        asset instead, labelled with IPTC trainedAlgorithmicMedia."""
        manifest = {
            "claim_generator_info": [{"name": "Deep-Guard", "version": "stage4"}],
            "title": title,
            "assertions": [
                {"label": "c2pa.soft-binding",
                 "data": {"alg": watermark_alg,
                          "blocks": [{"scope": {}, "value": f"{watermark_schema_code}*{watermark_bits}"}]}},
                {"label": LEDGER_ASSERTION,
                 "data": {"record_id": record_id, "creator": creator, "declared_origin": declared_origin,
                          "ledger_signer_key_id": ledger_key_id,
                          "about": "This copy is registered on a Deep-Guard provenance ledger. Check the record "
                                   "there: its fingerprints, the creator's signature and the block it was sealed in."}},
            ],
        }
        with self._lock:
            builder = c2pa.Builder(manifest)
            if declared_origin == "ai-generated" or parent is None:
                builder.set_intent(c2pa.C2paBuilderIntent.CREATE,
                                   c2pa.C2paDigitalSourceType.TRAINED_ALGORITHMIC_MEDIA if declared_origin == "ai-generated"
                                   else c2pa.C2paDigitalSourceType.EMPTY)
            else:
                builder.set_intent(c2pa.C2paBuilderIntent.EDIT)
                parent_bytes, parent_mime, parent_name = parent
                builder.add_ingredient({"title": parent_name, "relationship": "parentOf"}, parent_mime,
                                       io.BytesIO(parent_bytes))
            builder.add_action({"action": "c2pa.watermarked",
                                "softwareAgent": {"name": "Deep-Guard (TrustMark Q hidden ID)"}})
            signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(
                alg=c2pa.C2paSigningAlg.ED25519, sign_cert=self.signer_pem, private_key=self._signer_key_pem,
                ta_url=None))
            out = io.BytesIO()
            builder.sign(signer, mime, io.BytesIO(data), out)
            return out.getvalue()

    # ------------------------------------------------------------- reading
    def _store(self, data: bytes, anchors: Optional[str]) -> Optional[dict]:
        context = None
        if anchors:
            context = c2pa.Context(c2pa.Settings.from_dict({"trust": {"user_anchors": anchors},
                                                            "verify": {"verify_trust": True}}))
        reader = c2pa.Reader.try_create(None, io.BytesIO(data), context=context)
        if reader is None:
            return None
        try:
            return json.loads(reader.json())
        finally:
            reader.close()

    def read(self, data: bytes) -> dict:
        """What the file's Content Credentials say, checked twice: against the
        public trust lists, and against this server's own CA. Never raises."""
        try:
            with self._lock:
                public = self._store(data, self.public_anchors or None)
                if public is None:
                    return summarise(None)
                ours = self._store(data, self.ca_pem.decode("ascii"))
        except Exception as exc:  # noqa: BLE001 — a malformed manifest is a finding, not a crash
            return {"present": True, "verdict": "error", "headline": VERDICT_TEXT["error"],
                    "error": f"{exc.__class__.__name__}: {str(exc)[:200]}", "issues": []}
        return summarise(public, ours)


# ---------------------------------------------------------------------------
# Turning a manifest store into plain words
# ---------------------------------------------------------------------------

def _failure_codes(store: dict) -> list[str]:
    results = (store.get("validation_results") or {}).get("activeManifest") or {}
    codes = [f.get("code") for f in results.get("failure", []) if f.get("code")]
    codes += [s.get("code") for s in store.get("validation_status", []) or [] if s.get("code") and s.get("code") not in codes]
    return codes


def _is_integrity(code: str) -> bool:
    return code.endswith(".mismatch") or "Hash" in code or code.startswith("claimSignature")


def summarise(public: Optional[dict], ours: Optional[dict] = None) -> dict:
    if public is None:
        return {"present": False, "verdict": "none", "headline": VERDICT_TEXT["none"], "issues": []}
    active = (public.get("manifests") or {}).get(public.get("active_manifest") or "", {})
    sig = active.get("signature_info") or {}
    signer = sig.get("issuer") or sig.get("common_name") or "an unnamed signer"
    generator = active.get("claim_generator") or ", ".join(
        f"{g.get('name', '')} {g.get('version', '')}".strip() for g in active.get("claim_generator_info") or [])

    actions, source_type = [], None
    for assertion in active.get("assertions") or []:
        if assertion.get("label", "").startswith("c2pa.actions"):
            for action in (assertion.get("data") or {}).get("actions", []):
                short = _short_source_type(action.get("digitalSourceType", ""))
                agent = action.get("softwareAgent")
                agent = agent.get("name") if isinstance(agent, dict) else agent
                actions.append({"action": action.get("action"), "when": action.get("when"),
                                "software_agent": agent, "digital_source_type": short or None})
                if short and source_type is None:
                    source_type = short
    labels = {a.get("label"): a.get("data") for a in active.get("assertions") or []}
    soft = labels.get("c2pa.soft-binding")
    soft_binding = None
    if isinstance(soft, dict):
        value = ((soft.get("blocks") or [{}])[0] or {}).get("value", "")
        soft_binding = {"alg": soft.get("alg"), "value": value}
    deepguard = labels.get(LEDGER_ASSERTION) if isinstance(labels.get(LEDGER_ASSERTION), dict) else None

    codes = _failure_codes(public)
    public_state = public.get("validation_state")
    ours_state = (ours or {}).get("validation_state")
    integrity = [c for c in codes if _is_integrity(c)]
    if public_state == "Trusted":
        verdict = "trusted"
    elif integrity:
        verdict = "altered"
    elif ours_state == "Trusted":
        verdict = "ours"
    elif public_state == "Valid":
        verdict = "unknown"
    elif any(c in ("signingCredential.expired", "signingCredential.revoked") for c in codes):
        verdict = "expired"
    else:
        verdict = "problem"
    if verdict == "ours":
        codes = [c for c in codes if c != "signingCredential.untrusted"]  # untrusted only by the PUBLIC lists
    issues = [{"code": c, "text": ISSUE_TEXT.get(c, c)} for c in codes]
    first_issue = issues[0]["text"] if issues else "unknown reason"
    return {
        "present": True,
        "verdict": verdict,
        "headline": VERDICT_TEXT[verdict].format(signer=signer, issue=first_issue),
        "signer": signer,
        "signed_at": sig.get("time"),
        "algorithm": sig.get("alg"),
        "claim_generator": generator or None,
        "title": active.get("title"),
        "digital_source_type": source_type,
        "source_type_text": SOURCE_TYPES.get(source_type or "", source_type),
        "ai_declared": source_type in AI_SOURCE_TYPES,
        "capture_declared": source_type in CAPTURE_SOURCE_TYPES,
        "actions": actions,
        "ingredients": len(active.get("ingredients") or []),
        "manifests_in_history": len(public.get("manifests") or {}),
        "soft_binding": soft_binding,
        "deepguard": deepguard,
        "validation_state": public_state,
        "issues": issues,
    }


def to_evidence(summary: dict) -> dict:
    """Layer 0 of the trust check, in the shape evidence.py defines (the
    bouncer module is imported by ledger_app, so it is on the path)."""
    import evidence as ev  # deepguard-bouncer/app/evidence.py

    verdict = summary.get("verdict")
    limits = ["Credentials say who signed and what was declared; they do not prove the picture shows the truth."]
    if verdict == "none":
        item = ev.Evidence(0, "c2pa", "Content Credentials (C2PA)", ev.Status.NOT_FOUND, ev.PointsTo.NEITHER,
                           ev.Strength.NONE, summary["headline"],
                           limits=["Most files carry no credentials, and a re-save strips them."])
    elif verdict == "error":
        item = ev.Evidence(0, "c2pa", "Content Credentials (C2PA)", ev.Status.ERROR, ev.PointsTo.NEITHER,
                           ev.Strength.NONE, summary["headline"])
    else:
        points = (ev.PointsTo.AI if summary.get("ai_declared") else
                  ev.PointsTo.REAL if summary.get("capture_declared") else ev.PointsTo.NEITHER)
        if verdict == "trusted":
            strength = ev.Strength.PROOF
        else:
            strength = ev.Strength.STRONG
            if verdict in ("altered", "problem"):
                points = ev.PointsTo.NEITHER
            if verdict == "ours":
                limits.append("Signed by this server's own demo certificate authority, not one on a public trust list.")
            elif verdict == "unknown":
                limits.append("No trust list vouches for the signer, so anyone could have made these credentials.")
        item = ev.Evidence(0, "c2pa", "Content Credentials (C2PA)", ev.Status.FOUND, points, strength,
                           summary["headline"], details={k: summary.get(k) for k in (
                               "signer", "signed_at", "claim_generator", "digital_source_type", "validation_state")},
                           limits=limits)
    return item.validate().to_dict()
