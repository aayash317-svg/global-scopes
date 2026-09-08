"""
Pydantic Request & Response Schemas
Typed API contracts for all REST & WebSocket endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Audio Schemas ---
class AudioValidationResponse(BaseModel):
    valid: bool
    size_bytes: int
    duration_sec: float
    sample_rate: int
    snr_db: float
    clipping_ratio: float
    quality_rating: str
    is_usable: bool


class TelephonySimulationResponse(BaseModel):
    original_sample_rate: int
    telephony_sample_rate: int
    codec: str
    filtered_samples_count: int


# --- Detection Schemas ---
class SignalBreakdown(BaseModel):
    acoustic_score: float = Field(..., description="Phase jitter & waveform kurtosis anomaly score (0-1)")
    spectral_score: float = Field(..., description="Spectral flatness, rolloff, & HF loss score (0-1)")
    prosody_score: float = Field(..., description="Pitch F0 contour variation, micro-jitter & shimmer (0-1)")
    speaker_similarity: Optional[float] = Field(None, description="Cosine similarity to enrolled voiceprint (-1 to 1)")
    speaker_consistent: Optional[bool] = Field(None, description="Whether speaker matches enrolled KYC voiceprint")
    replay_probability: float = Field(..., description="Physical speaker playback probability (0-1)")


class DetectResponse(BaseModel):
    session_id: str
    has_speech: bool
    insufficient_speech: bool
    speech_ratio: float
    spoof_probability: float
    is_spoof: bool
    signals: SignalBreakdown
    model_source: str
    is_mock: bool


class FullAnalyzeResponse(BaseModel):
    session_id: str
    caller_id: str
    has_speech: bool
    insufficient_speech: bool
    spoof_probability: float
    risk_score: float
    risk_tier: str
    is_escalation_required: bool
    requires_step_up_mfa: bool
    safety_floor_applied: bool
    signals: SignalBreakdown
    reasons: List[str]
    audit_event_hash: str


# --- Risk Schemas ---
class RiskAssessRequest(BaseModel):
    session_id: Optional[str] = None
    caller_id: str = "caller_default"
    scenario: str = Field("routine_support", description="routine_support, high_value_transaction, privileged_access")
    call_origin: str = "VoIP"
    is_known_contact: bool = True
    transaction_value: float = 0.0
    fraud_history_score: float = 0.0
    acoustic_score: float
    spectral_score: float
    prosody_score: float
    speaker_similarity: Optional[float] = None
    replay_probability: float = 0.0


class RiskAssessResponse(BaseModel):
    session_id: Optional[str]
    risk_score: float
    risk_tier: str
    is_escalation_required: bool
    requires_step_up_mfa: bool
    safety_floor_applied: bool
    reasons: List[str]
    signals: Dict[str, float]


class RiskThresholdResponse(BaseModel):
    scenario: str
    low_max: float
    medium_max: float
    high_max: float
    medium_safety_floor: float
    high_safety_floor: float


# --- Verification Schemas ---
class EnrollVoiceprintRequest(BaseModel):
    user_id: str
    consent_granted: bool = True


class EnrollVoiceprintResponse(BaseModel):
    user_id: str
    enrolled: bool
    message: str
    embedding_dimension: int
    audit_event_hash: str


class ChallengeIssueRequest(BaseModel):
    session_id: str
    user_id: str
    challenge_type: str = "numeric_sequence"


class ChallengeIssueResponse(BaseModel):
    challenge_id: str
    token: str
    challenge_type: str
    prompt_text: str
    expires_in_sec: int
    expected_response: Optional[str] = None  # Provided for demo/testing convenience


class ChallengeVerifyRequest(BaseModel):
    token: str
    response_text: str


class ChallengeVerifyResponse(BaseModel):
    valid: bool
    message: str


class MFAIssueRequest(BaseModel):
    session_id: str
    user_id: str
    channel: str = "SMS"
    recipient: str = "+91-9999999999"


class MFAIssueResponse(BaseModel):
    token: str
    channel: str
    status: str
    expires_in_sec: int
    demo_code: Optional[str] = None


class MFAVerifyRequest(BaseModel):
    token: str
    code: str


class MFAVerifyResponse(BaseModel):
    valid: bool
    reason: str


class VerificationCallbackRequest(BaseModel):
    challenge_token: str
    verification_status: str = Field(..., description="PASSED, FAILED, EXPIRED")
    reviewer_notes: Optional[str] = None


class VerificationCallbackResponse(BaseModel):
    success: bool
    session_id: str
    verification_status: str
    session_status: str


# --- Privacy & Compliance Schemas ---
class ConsentRequest(BaseModel):
    user_id: str
    is_granted: bool = True


class ConsentResponse(BaseModel):
    user_id: str
    consent_granted: bool
    purpose: str
    timestamp: str


class ComplianceResponse(BaseModel):
    act: str = "Digital Personal Data Protection Act 2023 (DPDP Act 2023)"
    status: str = "Compliant"
    principles: Dict[str, str]
    retention_days: int
    biometric_encryption: str = "Fernet symmetric authenticated encryption"
    raw_audio_persisted: bool = False


# --- Audit Schemas ---
class AuditBlockItem(BaseModel):
    sequence_number: int
    timestamp: str
    event_type: str
    session_id: Optional[str]
    prev_hash: str
    event_hash: str
    payload: Dict[str, Any]


class AuditChainResponse(BaseModel):
    total_blocks: int
    blocks: List[AuditBlockItem]


class AuditVerifyResponse(BaseModel):
    is_valid: bool
    total_blocks: int
    head_hash: Optional[str]
    message: str
    tampered_block_sequence: Optional[int] = None
    tampered_field: Optional[str] = None
