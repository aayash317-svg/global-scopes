"""
Risk Engine Tests
Validates multi-signal risk calculation, context modulation, and the mandatory Safety-Floor Rule.
"""

import pytest

from backend.app.risk_engine.context import RiskContext
from backend.app.risk_engine.thresholds import RiskThresholdManager
from backend.app.risk_engine.scorer import RiskScorer
from backend.app.risk_engine.reasons import RiskReasonGenerator
from backend.app.inference.pipeline import UnifiedInferenceResult


def test_risk_context_multipliers():
    ctx_routine = RiskContext(scenario="routine_support")
    assert ctx_routine.get_contextual_multiplier() >= 1.0

    ctx_privileged = RiskContext(
        scenario="privileged_access",
        is_known_contact=False,
        transaction_value=1000000.0,
        fraud_history_score=0.8,
    )
    assert ctx_privileged.get_contextual_multiplier() > ctx_routine.get_contextual_multiplier()


def test_safety_floor_rule_enforcement():
    """
    Validates Master Reference Specification:
    A near-certain (>=90%) acoustic/spectral detection can no longer be suppressed by lenient context.
    >=90% single signal forces at least MEDIUM risk.
    >=97% single signal forces at least HIGH risk.
    """
    scorer = RiskScorer()

    # Lenient context: routine support, known contact, 0 transaction value
    lenient_ctx = RiskContext(scenario="routine_support", is_known_contact=True, transaction_value=0.0)

    # 1. Test >=97% single-signal confidence (e.g. 99.3% acoustic confidence)
    high_spoof_inf = UnifiedInferenceResult(
        has_speech=True,
        insufficient_speech=False,
        speech_ratio=1.0,
        acoustic_score=0.993,
        spectral_score=0.1,
        prosody_score=0.1,
        speaker_similarity=0.95,
        speaker_consistent=True,
        spoof_probability=0.993,
        is_spoof=True,
        replay_probability=0.0,
        is_replay=False,
        deepfake_model="AASIST",
        speaker_model="ECAPA",
        is_mock=True,
    )

    res_high = scorer.assess_risk(high_spoof_inf, lenient_ctx)
    assert res_high.safety_floor_applied is True
    assert res_high.risk_tier in ("HIGH", "CRITICAL")
    assert res_high.risk_score > 65.0  # Above medium threshold

    # 2. Test >=90% single-signal confidence (e.g. 92% spectral confidence)
    med_spoof_inf = UnifiedInferenceResult(
        has_speech=True,
        insufficient_speech=False,
        speech_ratio=1.0,
        acoustic_score=0.1,
        spectral_score=0.92,
        prosody_score=0.1,
        speaker_similarity=0.95,
        speaker_consistent=True,
        spoof_probability=0.92,
        is_spoof=True,
        replay_probability=0.0,
        is_replay=False,
        deepfake_model="AASIST",
        speaker_model="ECAPA",
        is_mock=True,
    )

    res_med = scorer.assess_risk(med_spoof_inf, lenient_ctx)
    assert res_med.safety_floor_applied is True
    assert res_med.risk_tier in ("MEDIUM", "HIGH", "CRITICAL")
    assert res_med.risk_score > 40.0  # Above low threshold


def test_insufficient_speech_state():
    scorer = RiskScorer()
    gated_inf = UnifiedInferenceResult(
        has_speech=False,
        insufficient_speech=True,
        speech_ratio=0.0,
        acoustic_score=0.0,
        spectral_score=0.0,
        prosody_score=0.0,
        speaker_similarity=None,
        speaker_consistent=None,
        spoof_probability=0.0,
        is_spoof=False,
        replay_probability=0.0,
        is_replay=False,
        deepfake_model="AASIST",
        speaker_model="ECAPA",
        is_mock=True,
    )

    res = scorer.assess_risk(gated_inf)
    assert res.risk_score == 0.0
    assert res.is_escalation_required is False
    assert any("INSUFFICIENT_SPEECH" in r for r in res.reasons)
