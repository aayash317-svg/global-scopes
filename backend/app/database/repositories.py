"""
Repository Layer for Database Access
Encapsulates all database queries and transactions.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, update, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import (
    CallSession,
    AudioMetadataRecord,
    DetectionResultRecord,
    SpeakerResultRecord,
    RiskResultRecord,
    VerificationRecord,
    ConsentRecord,
    AuditEventRecord,
)


class SessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, caller_id: str, origin: str = "VoIP", context_type: str = "routine_support") -> CallSession:
        session = CallSession(caller_id=caller_id, origin=origin, context_type=context_type)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get(self, session_id: str) -> Optional[CallSession]:
        stmt = select(CallSession).where(CallSession.id == session_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(self, session_id: str, status: str) -> Optional[CallSession]:
        session = await self.get(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now(timezone.utc)
            await self.db.commit()
            await self.db.refresh(session)
        return session

    async def list_recent(self, limit: int = 50) -> List[CallSession]:
        stmt = select(CallSession).order_by(desc(CallSession.created_at)).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


class AudioRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_metadata(
        self,
        session_id: str,
        sample_rate: int,
        channels: int,
        duration_seconds: float,
        format: str = "wav",
        snr_db: Optional[float] = None,
        clipping_ratio: Optional[float] = None,
    ) -> AudioMetadataRecord:
        record = AudioMetadataRecord(
            session_id=session_id,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            format=format,
            snr_db=snr_db,
            clipping_ratio=clipping_ratio,
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record


class DetectionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        session_id: str,
        window_index: int,
        spoof_probability: float,
        acoustic_score: float,
        spectral_score: float,
        prosody_score: float,
        replay_score: float,
        is_spoof: bool,
        model_source: str = "unified_pipeline",
    ) -> DetectionResultRecord:
        record = DetectionResultRecord(
            session_id=session_id,
            window_index=window_index,
            spoof_probability=spoof_probability,
            acoustic_score=acoustic_score,
            spectral_score=spectral_score,
            prosody_score=prosody_score,
            replay_score=replay_score,
            is_spoof=is_spoof,
            model_source=model_source,
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def get_by_session(self, session_id: str) -> List[DetectionResultRecord]:
        stmt = select(DetectionResultRecord).where(DetectionResultRecord.session_id == session_id).order_by(DetectionResultRecord.window_index)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


class SpeakerRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        session_id: str,
        enrolled_user_id: str,
        similarity_score: float,
        is_consistent: bool,
        threshold_used: float = 0.75,
    ) -> SpeakerResultRecord:
        record = SpeakerResultRecord(
            session_id=session_id,
            enrolled_user_id=enrolled_user_id,
            similarity_score=similarity_score,
            is_consistent=is_consistent,
            threshold_used=threshold_used,
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record


class RiskRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        session_id: str,
        risk_score: float,
        risk_level: str,
        floor_rule_applied: bool,
        reasons: List[str],
    ) -> RiskResultRecord:
        record = RiskResultRecord(
            session_id=session_id,
            risk_score=risk_score,
            risk_level=risk_level,
            floor_rule_applied=floor_rule_applied,
            reasons=json.dumps(reasons),
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def get_latest_by_session(self, session_id: str) -> Optional[RiskResultRecord]:
        stmt = (
            select(RiskResultRecord)
            .where(RiskResultRecord.session_id == session_id)
            .order_by(desc(RiskResultRecord.created_at))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()


class VerificationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        session_id: str,
        user_id: str,
        challenge_token: str,
        verification_type: str = "VOICE_CHALLENGE",
    ) -> VerificationRecord:
        record = VerificationRecord(
            session_id=session_id,
            user_id=user_id,
            challenge_token=challenge_token,
            verification_type=verification_type,
            status="PENDING",
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def get_by_token(self, token: str) -> Optional[VerificationRecord]:
        stmt = select(VerificationRecord).where(VerificationRecord.challenge_token == token)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(self, token: str, status: str) -> Optional[VerificationRecord]:
        record = await self.get_by_token(token)
        if record:
            record.status = status
            record.completed_at = datetime.now(timezone.utc)
            await self.db.commit()
            await self.db.refresh(record)
        return record


class PrivacyRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_consent(self, user_id: str) -> Optional[ConsentRecord]:
        stmt = select(ConsentRecord).where(ConsentRecord.user_id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_consent(
        self,
        user_id: str,
        is_granted: bool = True,
        encrypted_voiceprint: Optional[str] = None,
        purpose: str = "fraud_prevention_voice_verification",
    ) -> ConsentRecord:
        record = await self.get_consent(user_id)
        now = datetime.now(timezone.utc)
        if record:
            record.is_granted = is_granted
            if encrypted_voiceprint:
                record.encrypted_voiceprint = encrypted_voiceprint
            if not is_granted:
                record.revoked_at = now
            else:
                record.granted_at = now
                record.revoked_at = None
        else:
            record = ConsentRecord(
                user_id=user_id,
                purpose=purpose,
                is_granted=is_granted,
                encrypted_voiceprint=encrypted_voiceprint,
                granted_at=now,
            )
            self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def delete_user_biometrics(self, user_id: str) -> bool:
        record = await self.get_consent(user_id)
        if record:
            record.encrypted_voiceprint = None
            record.is_granted = False
            record.revoked_at = datetime.now(timezone.utc)
            await self.db.commit()
            return True
        return False


class AuditRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_latest(self) -> Optional[AuditEventRecord]:
        stmt = select(AuditEventRecord).order_by(desc(AuditEventRecord.sequence_number)).limit(1)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def append(
        self,
        event_type: str,
        prev_hash: str,
        event_hash: str,
        payload: dict,
        session_id: Optional[str] = None,
        timestamp_dt: Optional[datetime] = None,
    ) -> AuditEventRecord:
        latest = await self.get_latest()
        seq = (latest.sequence_number + 1) if latest else 1
        record = AuditEventRecord(
            sequence_number=seq,
            event_type=event_type,
            session_id=session_id,
            prev_hash=prev_hash,
            event_hash=event_hash,
            payload=json.dumps(payload),
            timestamp=timestamp_dt or datetime.now(timezone.utc),
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def get_all(self, limit: int = 100) -> List[AuditEventRecord]:
        stmt = select(AuditEventRecord).order_by(AuditEventRecord.sequence_number).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
