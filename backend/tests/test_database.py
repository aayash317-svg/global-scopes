"""
Database Repositories Tests
Validates CRUD transactions across all relational persistence entities.
"""

import pytest

from backend.app.database.repositories import (
    SessionRepository,
    AudioRepository,
    DetectionRepository,
    SpeakerRepository,
    RiskRepository,
    VerificationRepository,
    PrivacyRepository,
)


@pytest.mark.asyncio
async def test_session_and_audio_repositories(db_session):
    s_repo = SessionRepository(db_session)
    session = await s_repo.create(caller_id="test_caller_01", origin="VoIP")
    assert session.id is not None
    assert session.status == "ACTIVE"

    # Audio metadata
    a_repo = AudioRepository(db_session)
    audio_rec = await a_repo.create_metadata(
        session_id=session.id,
        sample_rate=16000,
        channels=1,
        duration_seconds=3.5,
        snr_db=25.4,
    )
    assert audio_rec.session_id == session.id


@pytest.mark.asyncio
async def test_detection_and_risk_repositories(db_session):
    s_repo = SessionRepository(db_session)
    session = await s_repo.create(caller_id="test_caller_02")

    d_repo = DetectionRepository(db_session)
    det = await d_repo.create(
        session_id=session.id,
        window_index=0,
        spoof_probability=0.88,
        acoustic_score=0.9,
        spectral_score=0.85,
        prosody_score=0.8,
        replay_score=0.1,
        is_spoof=True,
    )
    assert det.is_spoof is True

    r_repo = RiskRepository(db_session)
    risk = await r_repo.create(
        session_id=session.id,
        risk_score=78.5,
        risk_level="HIGH",
        floor_rule_applied=True,
        reasons=["Phase anomaly", "Floor rule activated"],
    )
    assert risk.risk_level == "HIGH"

    fetched = await r_repo.get_latest_by_session(session.id)
    assert fetched.risk_score == 78.5


@pytest.mark.asyncio
async def test_verification_repository(db_session):
    s_repo = SessionRepository(db_session)
    session = await s_repo.create(caller_id="caller_verif")

    v_repo = VerificationRepository(db_session)
    rec = await v_repo.create(
        session_id=session.id,
        user_id="caller_verif",
        challenge_token="token_abc_123",
        verification_type="VOICE_CHALLENGE",
    )
    assert rec.status == "PENDING"

    updated = await v_repo.update_status("token_abc_123", "PASSED")
    assert updated.status == "PASSED"
