"""
Detection API Routes
Handles Layer 1 multi-signal voice authenticity inference and unified analysis.
Thin route controllers delegating to inference pipeline and repositories.
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from backend.app.database.session import get_db
from backend.app.database.repositories import (
    SessionRepository,
    AudioRepository,
    DetectionRepository,
    SpeakerRepository,
    RiskRepository,
    PrivacyRepository,
)
from backend.app.api.schemas import DetectResponse, FullAnalyzeResponse, SignalBreakdown
from backend.app.audio.preprocessing import load_audio, validate_audio_input
from backend.app.audio.quality import AudioQualityAnalyzer
from backend.app.privacy.encryption import BiometricEncryptionService
from backend.app.privacy.deletion import DataDeletionManager
from backend.app.inference.pipeline import UnifiedInferencePipeline
from backend.app.risk_engine.scorer import RiskScorer
from backend.app.risk_engine.context import RiskContext
from backend.app.audit.hash_chain import HashChainService
from backend.app.audit.events import AuditEventType

router = APIRouter()
pipeline = UnifiedInferencePipeline()
scorer = RiskScorer()
quality_analyzer = AudioQualityAnalyzer()
crypto_service = BiometricEncryptionService()


@router.post("/detect", response_model=DetectResponse, summary="Layer 1 voice authenticity inference")
async def detect_spoof(
    file: UploadFile = File(...),
    caller_id: str = Form("caller_anonymous"),
    session_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Executes Layer 1 spoof, acoustic, spectral, prosody, and replay analysis."""
    try:
        content = await file.read()
        validate_audio_input(content)
        audio_data, sr = load_audio(content, target_sr=16000)

        # Ensure session exists
        session_repo = SessionRepository(db)
        if not session_id:
            call_session = await session_repo.create(caller_id=caller_id)
            session_id = call_session.id

        # Check for enrolled voiceprint for this caller
        privacy_repo = PrivacyRepository(db)
        consent = await privacy_repo.get_consent(caller_id)
        enrolled_emb = None
        if consent and consent.encrypted_voiceprint:
            enrolled_emb = crypto_service.decrypt_embedding(consent.encrypted_voiceprint)

        # Run inference pipeline
        inf = pipeline.process_window(audio_data, enrolled_embedding=enrolled_emb, sr=sr)

        # Persist detection record via repository
        det_repo = DetectionRepository(db)
        await det_repo.create(
            session_id=session_id,
            window_index=0,
            spoof_probability=inf.spoof_probability,
            acoustic_score=inf.acoustic_score,
            spectral_score=inf.spectral_score,
            prosody_score=inf.prosody_score,
            replay_score=inf.replay_probability,
            is_spoof=inf.is_spoof,
            model_source=inf.deepfake_model,
        )

        # Record speaker result if verified
        if inf.speaker_similarity is not None:
            spk_repo = SpeakerRepository(db)
            await spk_repo.create(
                session_id=session_id,
                enrolled_user_id=caller_id,
                similarity_score=inf.speaker_similarity,
                is_consistent=inf.speaker_consistent,
            )

        # Memory zeroization (Data minimisation guarantee)
        DataDeletionManager.zeroize_audio_buffer(audio_data)

        return DetectResponse(
            session_id=session_id,
            has_speech=inf.has_speech,
            insufficient_speech=inf.insufficient_speech,
            speech_ratio=inf.speech_ratio,
            spoof_probability=inf.spoof_probability,
            is_spoof=inf.is_spoof,
            signals=SignalBreakdown(
                acoustic_score=inf.acoustic_score,
                spectral_score=inf.spectral_score,
                prosody_score=inf.prosody_score,
                speaker_similarity=inf.speaker_similarity,
                speaker_consistent=inf.speaker_consistent,
                replay_probability=inf.replay_probability,
            ),
            model_source=inf.deepfake_model,
            is_mock=inf.is_mock,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze", response_model=FullAnalyzeResponse, summary="End-to-end multi-layer voice analysis")
async def analyze_full_pipeline(
    file: UploadFile = File(...),
    caller_id: str = Form("caller_default"),
    scenario: str = Form("routine_support"),
    call_origin: str = Form("VoIP"),
    transaction_value: float = Form(0.0),
    is_known_contact: bool = Form(True),
    fraud_history_score: float = Form(0.0),
    session_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Complete 5-layer pipeline execution:
    Audio chunk -> VAD -> 4 Layer-1 signals -> Risk scoring & safety floor ->
    Alert generation -> Cryptographic hash-chain audit block.
    """
    try:
        content = await file.read()
        validate_audio_input(content)
        audio_data, sr = load_audio(content, target_sr=16000)

        # Audio metadata analysis
        q_metrics = quality_analyzer.analyze(audio_data, sr)

        # Session & Repositories
        session_repo = SessionRepository(db)
        if not session_id:
            call_session = await session_repo.create(caller_id=caller_id, origin=call_origin, context_type=scenario)
            session_id = call_session.id

        audio_repo = AudioRepository(db)
        await audio_repo.create_metadata(
            session_id=session_id,
            sample_rate=sr,
            channels=1,
            duration_seconds=q_metrics["duration_sec"],
            snr_db=q_metrics["snr_db"],
            clipping_ratio=q_metrics["clipping_ratio"],
        )

        # Retrieve encrypted voiceprint if exists
        privacy_repo = PrivacyRepository(db)
        consent = await privacy_repo.get_consent(caller_id)
        enrolled_emb = None
        if consent and consent.encrypted_voiceprint:
            enrolled_emb = crypto_service.decrypt_embedding(consent.encrypted_voiceprint)

        # Layer 1: Multi-signal inference
        inf = pipeline.process_window(audio_data, enrolled_embedding=enrolled_emb, sr=sr)

        # Layer 2: Contextual Risk Scoring + Safety Floor Rule
        context = RiskContext(
            scenario=scenario,
            call_origin=call_origin,
            is_known_contact=is_known_contact,
            transaction_value=transaction_value,
            fraud_history_score=fraud_history_score,
        )
        risk = scorer.assess_risk(inf, context)

        # Persist Detections & Risk
        det_repo = DetectionRepository(db)
        await det_repo.create(
            session_id=session_id,
            window_index=0,
            spoof_probability=inf.spoof_probability,
            acoustic_score=inf.acoustic_score,
            spectral_score=inf.spectral_score,
            prosody_score=inf.prosody_score,
            replay_score=inf.replay_probability,
            is_spoof=inf.is_spoof,
            model_source=inf.deepfake_model,
        )

        risk_repo = RiskRepository(db)
        await risk_repo.create(
            session_id=session_id,
            risk_score=risk.risk_score,
            risk_level=risk.risk_tier,
            floor_rule_applied=risk.safety_floor_applied,
            reasons=risk.reasons,
        )

        # Layer 4: Append security event to Tamper-Evident Hash Chain
        hash_svc = HashChainService(db)
        audit_block = await hash_svc.record_event(
            event_type=AuditEventType.RISK_EVALUATED.value,
            payload={
                "caller_id": caller_id,
                "risk_score": risk.risk_score,
                "risk_tier": risk.risk_tier,
                "spoof_probability": inf.spoof_probability,
                "floor_applied": risk.safety_floor_applied,
                "reasons_count": len(risk.reasons),
            },
            session_id=session_id,
        )

        # Update Session state if High Risk
        if risk.is_escalation_required:
            await session_repo.update_status(session_id, "FLAGGED")

        # Memory zeroization
        DataDeletionManager.zeroize_audio_buffer(audio_data)

        return FullAnalyzeResponse(
            session_id=session_id,
            caller_id=caller_id,
            has_speech=inf.has_speech,
            insufficient_speech=inf.insufficient_speech,
            spoof_probability=inf.spoof_probability,
            risk_score=risk.risk_score,
            risk_tier=risk.risk_tier,
            is_escalation_required=risk.is_escalation_required,
            requires_step_up_mfa=risk.requires_step_up_mfa,
            safety_floor_applied=risk.safety_floor_applied,
            signals=SignalBreakdown(
                acoustic_score=inf.acoustic_score,
                spectral_score=inf.spectral_score,
                prosody_score=inf.prosody_score,
                speaker_similarity=inf.speaker_similarity,
                speaker_consistent=inf.speaker_consistent,
                replay_probability=inf.replay_probability,
            ),
            reasons=risk.reasons,
            audit_event_hash=audit_block.event_hash,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
