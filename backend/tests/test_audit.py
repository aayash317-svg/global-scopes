"""
Tamper-Evident Hash-Chain Audit Trail Tests
Theme Alignment: Blockchain & Cybersecurity
Directly verifies cryptographic immutability and deterministic tamper detection.
"""

import pytest
import json

from backend.app.audit.hash_chain import HashChainService
from backend.app.audit.events import AuditEventType
from backend.app.database.repositories import AuditRepository


@pytest.mark.asyncio
async def test_hash_chain_append_and_verify(db_session):
    svc = HashChainService(db_session)

    # 1. Record multiple security events
    e1 = await svc.record_event(
        event_type=AuditEventType.CALL_STARTED.value,
        payload={"caller_id": "user_101", "origin": "VoIP"},
    )
    e2 = await svc.record_event(
        event_type=AuditEventType.RISK_EVALUATED.value,
        payload={"risk_score": 75.0, "tier": "HIGH"},
    )
    e3 = await svc.record_event(
        event_type=AuditEventType.ALERT_TRIGGERED.value,
        payload={"channel": "SMS", "action": "STEP_UP_OTP"},
    )

    # 2. Verify clean chain integrity
    res = await svc.verify_chain()
    assert res.is_valid is True
    assert res.total_blocks == 4  # Genesis + 3 events
    assert res.head_hash == e3.event_hash


@pytest.mark.asyncio
async def test_hash_chain_tamper_detection(db_session):
    """
    Simulates malicious modification of a historical risk record
    (e.g., attacker edits a historical 'CRITICAL' risk record to read 'LOW')
    and asserts that verify_chain() immediately flags tampering.
    """
    svc = HashChainService(db_session)

    # Append events
    await svc.record_event(
        event_type=AuditEventType.CALL_STARTED.value,
        payload={"caller_id": "victim_cxo"},
    )
    risk_event = await svc.record_event(
        event_type=AuditEventType.RISK_EVALUATED.value,
        payload={"risk_score": 98.5, "tier": "CRITICAL", "spoof_detected": True},
    )
    await svc.record_event(
        event_type=AuditEventType.ALERT_TRIGGERED.value,
        payload={"alert": "SUPERVISOR_ESCALATION"},
    )

    # Clean check
    clean_res = await svc.verify_chain()
    assert clean_res.is_valid is True

    # Deliberate tampering: alter the risk_event payload in the database
    risk_event.payload = json.dumps({"risk_score": 10.0, "tier": "LOW", "spoof_detected": False})
    await db_session.commit()

    # Verify chain again: must flag tamper
    tamper_res = await svc.verify_chain()
    assert tamper_res.is_valid is False
    assert tamper_res.tampered_block_sequence == risk_event.sequence_number
    assert tamper_res.tampered_field == "event_hash"
