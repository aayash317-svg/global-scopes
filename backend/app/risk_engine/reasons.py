"""
Explainable Risk Reasons Generator
Maps detected biometric anomalies and contextual risks to human-readable explanations.
"""

from typing import List
from backend.app.inference.pipeline import UnifiedInferenceResult
from backend.app.risk_engine.context import RiskContext


class RiskReasonGenerator:
    """Generates human-auditable risk indicators and explanation codes."""

    @staticmethod
    def generate_reasons(
        inference: UnifiedInferenceResult,
        context: RiskContext,
        floor_applied: bool,
        peak_signal_val: float,
    ) -> List[str]:
        reasons: List[str] = []

        # Layer 1 Signal Explanations
        if inference.acoustic_score >= 0.65:
            reasons.append(
                f"ACOUSTIC_PHASE_ANOMALY: Phase jitter ({inference.acoustic_score:.2f}) indicates waveform synthesis artifacts."
            )

        if inference.spectral_score >= 0.65:
            reasons.append(
                f"SPECTRAL_SYNTHESIS_SIGNATURE: Unnatural spectral flatness ({inference.spectral_score:.2f}) and high-frequency loss."
            )

        if inference.prosody_score >= 0.65:
            reasons.append(
                f"PROSODY_ROBOTIC_MONOTONE: Pitch contour flatness ({inference.prosody_score:.2f}) reflects neural speech synthesis."
            )

        if inference.speaker_consistent is False and inference.speaker_similarity is not None:
            reasons.append(
                f"SPEAKER_VOICEPRINT_MISMATCH: Caller voice does not match enrolled biometric voiceprint (similarity: {inference.speaker_similarity:.2f})."
            )

        if inference.is_replay or inference.replay_probability >= 0.65:
            reasons.append(
                f"PHYSICAL_REPLAY_DETECTED: Acoustic playback transducer signatures and smearing detected ({inference.replay_probability:.2f})."
            )

        # Safety Floor Explanation
        if floor_applied:
            reasons.append(
                f"SAFETY_FLOOR_TRIGGERED: Mandatory safety floor activated due to near-certain single-signal confidence ({peak_signal_val * 100:.1f}%)."
            )

        # Contextual Threat Explanations
        if context.transaction_value > 100000.0:
            reasons.append(
                f"HIGH_VALUE_EXPOSURE: Elevated financial transaction exposure (Value: ₹{context.transaction_value:,.2f})."
            )

        if not context.is_known_contact:
            reasons.append("UNVERIFIED_CALLER: Inbound call from unknown or unverified contact identifier.")

        if context.fraud_history_score > 0.4:
            reasons.append(f"PRIOR_FRAUD_HISTORY: Associated account has prior fraud incidents (Risk index: {context.fraud_history_score:.2f}).")

        if context.scenario == "privileged_access":
            reasons.append("PRIVILEGED_ACCESS_TARGET: Operation targets sensitive credential reset or administrative authorization.")

        return reasons
