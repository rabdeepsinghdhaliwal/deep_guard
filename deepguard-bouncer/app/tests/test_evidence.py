"""
Tests evidence.py -- the one answer format every layer of the trust check
will use. The rules under test are the ones that stop the report from
saying something untrue.
"""
import pytest

from evidence import Evidence, PointsTo, Status, Strength, disagreements, strongest_first


def make(**overrides):
    base = dict(layer=2, check="efficientnet", title="Our AI-image detector", status=Status.FOUND,
                points_to=PointsTo.AI, strength=Strength.INDICATION, claim="Looks AI-generated (91%).",
                confidence=0.91)
    base.update(overrides)
    return Evidence(**base)


def test_a_normal_detector_finding_is_valid():
    assert make().validate()


def test_nothing_found_cannot_carry_weight():
    with pytest.raises(ValueError, match="cannot carry"):
        make(status=Status.NOT_FOUND, strength=Strength.STRONG, points_to=PointsTo.NEITHER).validate()


def test_nothing_found_cannot_point_anywhere():
    with pytest.raises(ValueError, match="only a 'found' result"):
        make(status=Status.NOT_RUN, strength=Strength.NONE, points_to=PointsTo.AI).validate()


def test_only_a_verified_signature_is_proof():
    with pytest.raises(ValueError, match="proof"):
        make(strength=Strength.PROOF).validate()
    assert make(layer=0, check="c2pa", strength=Strength.PROOF, confidence=None,
                claim="Signed by OpenAI: made with gpt-image-1.").validate()


def test_a_statistical_indication_needs_its_confidence():
    with pytest.raises(ValueError, match="confidence"):
        make(confidence=None).validate()
    with pytest.raises(ValueError, match="between 0 and 1"):
        make(confidence=1.5).validate()


def test_round_trip_through_json_shape():
    e = make(details={"p_ai_raw": 0.97}, limits=["a guess, not proof"])
    d = e.to_dict()
    assert d["status"] == "found" and d["layer_name"] == "Detectors"
    assert Evidence.from_dict(d) == e


def test_strongest_first_orders_by_strength_then_layer():
    sig = make(layer=0, check="c2pa", strength=Strength.PROOF, confidence=None, claim="Signed.")
    wm = make(layer=1, check="trustmark", strength=Strength.STRONG, confidence=None, claim="Watermark found.")
    det = make()
    none = make(layer=3, check="registry", status=Status.NOT_FOUND, strength=Strength.NONE,
                points_to=PointsTo.NEITHER, confidence=None, claim="No registered copy.")
    assert [e.check for e in strongest_first([none, det, wm, sig])] == ["c2pa", "trustmark", "efficientnet", "registry"]


def test_disagreements_are_found_not_averaged():
    camera = make(layer=0, check="c2pa", strength=Strength.PROOF, points_to=PointsTo.REAL, confidence=None,
                  claim="Signed by a camera at capture.")
    wm = make(layer=1, check="synthid", strength=Strength.STRONG, confidence=None, claim="AI watermark found.")
    pairs = disagreements([camera, wm, make()])
    assert (camera, wm) in pairs and len(pairs) == 2
