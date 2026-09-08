"""
Risk Engine API Routes
Endpoints for risk assessment, active threshold inspection, and session risk queries.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.session import get_db
from backend.app.database.repositories import RiskRepository
from backend.app.api.schemas import RiskAssessRequest, RiskAssessResponse, RiskThresholdResponse
from backend.app.risk_engine.scorer import RiskScorer
from backend.app.risk_engine.context import RiskContext
from backend.app.risk_engine.thresholds import RiskThresholdManager
from backend.app.inference.pipeline import UnifiedInferenceResult

router = APIRouter()
scorer = RiskScorer()
threshold_manager = RiskThresholdManager()


@router.post("/assess", response_model=RiskAssessResponse, summary="Standalone risk assessment calculation")
async def assess_risk_scores(request: RiskAssessRequest, db: AsyncSession = Depends(get_db)):
    """Computes dynamic risk score and enforces safety floor rule."""
    try:
        inf = UnifiedInferenceResult(
            has_speech=True,
            insufficient_speech=False,
            speech_ratio=1.0,
            acoustic_score=request.acoustic_score,
            spectral_score=request.spectral_score,
            prosody_score=request.prosody_score,
            speaker_similarity=request.speaker_similarity,
            speaker_consistent=request.speaker_similarity >= 0.75 if request.speaker_similarity is not None else None,
            spoof_probability=max(request.acoustic_score, request.spectral_score),
            is_spoof=max(request.acoustic_score, request.spectral_score) >= 0.5,
            replay_probability=request.replay_probability,
            is_replay=request.replay_probability >= 0.65,
            deepfake_model="API_Inference",
            speaker_model="ECAPA",
            is_mock=False,
        )

        ctx = RiskContext(
            scenario=request.scenario,
            call_origin=request.call_origin,
            is_known_contact=request.is_known_contact,
            transaction_value=request.transaction_value,
            fraud_history_score=request.fraud_history_score,
        )

        res = scorer.assess_risk(inf, ctx)

        # Save to database if session_id provided
        if request.session_id:
            risk_repo = RiskRepository(db)
            await risk_repo.create(
                session_id=request.session_id,
                risk_score=res.risk_score,
                risk_level=res.risk_tier,
                floor_rule_applied=res.safety_floor_applied,
                reasons=res.reasons,
            )

        return RiskAssessResponse(
            session_id=request.session_id,
            risk_score=res.risk_score,
            risk_tier=res.risk_tier,
            is_escalation_required=res.is_escalation_required,
            requires_step_up_mfa=res.requires_step_up_mfa,
            safety_floor_applied=res.safety_floor_applied,
            reasons=res.reasons,
            signals=res.signal_breakdown,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/thresholds", response_model=List[RiskThresholdResponse], summary="Retrieve active risk scenario thresholds")
async def get_thresholds():
    """Returns configured thresholds and safety floor levels across scenarios."""
    scenarios = ["routine_support", "high_value_transaction", "privileged_access"]
    results = []
    for sc in scenarios:
        th = threshold_manager.get_thresholds(sc)
        results.append(
            RiskThresholdResponse(
                scenario=sc,
                low_max=th.low_max,
                medium_max=th.medium_max,
                high_max=th.high_max,
                medium_safety_floor=th.medium_safety_floor,
                high_safety_floor=th.high_safety_floor,
            )
        )
    return results


@router.get("/session/{session_id}", summary="Retrieve latest risk record for session")
async def get_session_risk(session_id: str, db: AsyncSession = Depends(get_db)):
    """Fetches latest risk evaluation stored for a given call session."""
    risk_repo = RiskRepository(db)
    record = await risk_repo.get_latest_by_session(session_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No risk assessment found for session {session_id}")

    import json
    return {
        "session_id": record.session_id,
        "risk_score": record.risk_score,
        "risk_level": record.risk_level,
        "floor_rule_applied": record.floor_rule_applied,
        "reasons": json.loads(record.reasons) if isinstance(record.reasons, str) else record.reasons,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }
