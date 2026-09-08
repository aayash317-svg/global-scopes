"""
SQLAlchemy ORM Models
Defines relational entities for sessions, audio metadata, ML detections,
risk calculations, verification workflows, privacy/consent, and tamper-evident audit.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.database.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CallSession(Base):
    """Represents a monitored call / communication session."""
    __tablename__ = "call_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    caller_id = Column(String(128), nullable=False, index=True)
    status = Column(String(32), default="ACTIVE", index=True)  # ACTIVE, COMPLETED, FLAGGED, TERMINATED
    origin = Column(String(64), default="VoIP")  # VoIP, PSTN, SIP, WebRTC
    context_type = Column(String(64), default="routine_support")  # routine_support, high_value_transaction, privileged_access
    created_at = Column(DateTime(timezone=True), default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), default=get_utc_now, onupdate=get_utc_now)

    # Relationships
    audio_metadata = relationship("AudioMetadataRecord", back_populates="session", cascade="all, delete-orphan")
    detections = relationship("DetectionResultRecord", back_populates="session", cascade="all, delete-orphan")
    speaker_results = relationship("SpeakerResultRecord", back_populates="session", cascade="all, delete-orphan")
    risk_results = relationship("RiskResultRecord", back_populates="session", cascade="all, delete-orphan")
    verifications = relationship("VerificationRecord", back_populates="session", cascade="all, delete-orphan")


class AudioMetadataRecord(Base):
    """Audio quality and capture parameters (no raw audio stored)."""
    __tablename__ = "audio_metadata"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("call_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    sample_rate = Column(Integer, default=16000)
    channels = Column(Integer, default=1)
    duration_seconds = Column(Float, nullable=False)
    format = Column(String(16), default="wav")
    snr_db = Column(Float, nullable=True)
    clipping_ratio = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    session = relationship("CallSession", back_populates="audio_metadata")


class DetectionResultRecord(Base):
    """Layer 1 spoof detection inference output per audio window."""
    __tablename__ = "detection_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("call_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    window_index = Column(Integer, default=0)
    spoof_probability = Column(Float, nullable=False)
    acoustic_score = Column(Float, default=0.0)
    spectral_score = Column(Float, default=0.0)
    prosody_score = Column(Float, default=0.0)
    replay_score = Column(Float, default=0.0)
    is_spoof = Column(Boolean, default=False)
    model_source = Column(String(64), default="unified_pipeline")
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    session = relationship("CallSession", back_populates="detections")


class SpeakerResultRecord(Base):
    """Layer 1 cross-session speaker verification result."""
    __tablename__ = "speaker_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("call_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    enrolled_user_id = Column(String(128), nullable=False)
    similarity_score = Column(Float, nullable=False)
    is_consistent = Column(Boolean, default=True)
    threshold_used = Column(Float, default=0.75)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    session = relationship("CallSession", back_populates="speaker_results")


class RiskResultRecord(Base):
    """Layer 2 real-time risk assessment."""
    __tablename__ = "risk_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("call_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)  # 0 to 100
    risk_level = Column(String(16), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    floor_rule_applied = Column(Boolean, default=False)
    reasons = Column(Text, default="[]")  # JSON encoded list of reasons
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    session = relationship("CallSession", back_populates="risk_results")


class VerificationRecord(Base):
    """Layer 3 step-up multi-factor authentication or voice challenge."""
    __tablename__ = "verification_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("call_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(128), nullable=False)
    verification_type = Column(String(32), default="VOICE_CHALLENGE")  # VOICE_CHALLENGE, OTP, SUPERVISOR_CALLBACK
    challenge_token = Column(String(64), nullable=False)
    status = Column(String(16), default="PENDING")  # PENDING, PASSED, FAILED, EXPIRED
    created_at = Column(DateTime(timezone=True), default=get_utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("CallSession", back_populates="verifications")


class ConsentRecord(Base):
    """Layer 4 DPDP Act 2023 consent record and encrypted voiceprint."""
    __tablename__ = "consent_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(128), unique=True, nullable=False, index=True)
    purpose = Column(String(128), default="fraud_prevention_voice_verification")
    is_granted = Column(Boolean, default=True)
    encrypted_voiceprint = Column(Text, nullable=True)  # Fernet-encrypted embedding string
    granted_at = Column(DateTime(timezone=True), default=get_utc_now)
    revoked_at = Column(DateTime(timezone=True), nullable=True)


class AuditEventRecord(Base):
    """Layer 4 tamper-evident hash-chain audit log."""
    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    sequence_number = Column(Integer, unique=True, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=get_utc_now)
    event_type = Column(String(64), nullable=False, index=True)
    session_id = Column(String(36), nullable=True, index=True)
    prev_hash = Column(String(64), nullable=False)
    event_hash = Column(String(64), nullable=False, index=True)
    payload = Column(Text, nullable=False)  # JSON metadata only
