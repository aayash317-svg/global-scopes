"""
Real-Time Risk Scorer Subsystem
Implements Layer 2 risk scoring:
- Blends 4 Layer 1 signals with contextual risk multipliers.
- Enforces mandatory Safety-Floor rule to prevent context dilution of high-confidence spoofs.
- Assigns actionable risk tiers (LOW, MEDIUM, HIGH, CRITICAL).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

from backend.app.inference.pipeline import UnifiedInferenceResult
from backend.app.risk_engine.context import RiskContext
from backend.app.risk_engine.thresholds import RiskThresholdManager, ScenarioThresholds
from backend.app.risk_engine.reasons import RiskReasonGenerator


@dataclass
class RiskAssessmentResult:
    risk_score: float  # 0.0 to 100.0
    risk_tier: str     # LOW, MEDIUM, HIGH, CRITICAL
    is_escalation_required: bool
    requires_step_up_mfa: bool
    safety_floor_applied: bool
    scenario: str
    reasons: List[str]
    signal_breakdown: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


class RiskScorer:
    """
    Computes comprehensive 0-100 risk score and assigns action tiers.
    Applies the SIH26104 safety floor rule ensuring >=90% single-signal confidence
    forces at least MEDIUM tier and >=97% forces at least HIGH tier.
    """

    def __init__(self, threshold_manager: RiskThresholdManager = None):
        self.threshold_manager = threshold_manager or RiskThresholdManager()

    def assess_risk(
        self,
        inference: UnifiedInferenceResult,
        context: RiskContext = None,
    ) -> RiskAssessmentResult:
        context = context or RiskContext()
        thresholds: ScenarioThresholds = self.threshold_manager.get_thresholds(context.scenario)

        # If audio had insufficient vocal content (silence/noise gate)
        if inference.insufficient_speech:
            return RiskAssessmentResult(
                risk_score=0.0,
                risk_tier="LOW",
                is_escalation_required=False,
                requires_step_up_mfa=False,
                safety_floor_applied=False,
                scenario=context.scenario,
                reasons=["INSUFFICIENT_SPEECH: Window gated by VAD due to lack of distinct speech."],
                signal_breakdown={
                    "acoustic": 0.0,
                    "spectral": 0.0,
                    "prosody": 0.0,
                    "speaker_mismatch": 0.0,
                    "replay": 0.0,
                },
            )

        # 1. Base Biometric Signal Scoring (0.0 to 1.0)
        ac = inference.acoustic_score
        sp = inference.spectral_score
        pr = inference.prosody_score
        rep = inference.replay_probability

        # Speaker consistency signal:
        # If enrolled voiceprint is present and mismatch detected, add penalty
        if inference.speaker_similarity is not None:
            # Low similarity means high risk of impersonation
            speaker_mismatch = float(np.clip(1.0 - inference.speaker_similarity, 0.0, 1.0))
            # Weight distribution with speaker verification
            raw_biometric_score = (
                0.25 * ac +
                0.25 * sp +
                0.20 * pr +
                0.20 * speaker_mismatch +
                0.10 * rep
            )
        else:
            # Limitation 1 fallback: Caller not enrolled yet -> re-normalize weights
            speaker_mismatch = 0.0
            raw_biometric_score = (
                0.35 * ac +
                0.35 * sp +
                0.20 * pr +
                0.10 * rep
            )

        # Convert to 0 - 100 scale
        base_score = raw_biometric_score * 100.0

        # 2. Contextual Modulation
        context_mult = context.get_contextual_multiplier()
        modulated_score = base_score * context_mult

        # 3. Mandatory Safety Floor Rule
        # Peak single-signal confidence across Layer 1 spoof signals
        peak_single_signal = max(ac, sp, pr, inference.spoof_probability)
        floor_applied = False

        if peak_single_signal >= thresholds.high_safety_floor:
            # >= 97% confidence: must be at least HIGH risk tier
            if modulated_score <= thresholds.medium_max:
                modulated_score = thresholds.medium_max + 5.0
                floor_applied = True
        elif peak_single_signal >= thresholds.medium_safety_floor:
            # >= 90% confidence: must be at least MEDIUM risk tier
            if modulated_score <= thresholds.low_max:
                modulated_score = thresholds.low_max + 5.0
                floor_applied = True

        # Clamp score between 0.0 and 100.0
        final_score = float(round(np.clip(modulated_score, 0.0, 100.0), 2))

        # 4. Assign Risk Tier
        if final_score <= thresholds.low_max:
            tier = "LOW"
        elif final_score <= thresholds.medium_max:
            tier = "MEDIUM"
        elif final_score <= thresholds.high_max:
            tier = "HIGH"
        else:
            tier = "CRITICAL"

        is_escalation = tier in ("HIGH", "CRITICAL")
        requires_mfa = tier in ("MEDIUM", "HIGH", "CRITICAL")

        # 5. Generate Reasons
        reasons = RiskReasonGenerator.generate_reasons(
            inference=inference,
            context=context,
            floor_applied=floor_applied,
            peak_signal_val=peak_single_signal,
        )

        return RiskAssessmentResult(
            risk_score=final_score,
            risk_tier=tier,
            is_escalation_required=is_escalation,
            requires_step_up_mfa=requires_mfa,
            safety_floor_applied=floor_applied,
            scenario=context.scenario,
            reasons=reasons,
            signal_breakdown={
                "acoustic": round(ac, 4),
                "spectral": round(sp, 4),
                "prosody": round(pr, 4),
                "speaker_mismatch": round(speaker_mismatch, 4),
                "replay": round(rep, 4),
            },
            details={
                "context_multiplier": context_mult,
                "peak_single_signal": round(peak_single_signal, 4),
                "thresholds": {
                    "low_max": thresholds.low_max,
                    "medium_max": thresholds.medium_max,
                    "high_max": thresholds.high_max,
                },
            },
        )
