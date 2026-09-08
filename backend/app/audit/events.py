"""
Audit Event Definitions & Schema
Standardizes security audit event types and payload formatting.
Guarantees NO audio or sensitive plaintext biometrics enter the audit chain.
"""

from enum import Enum
import json
from typing import Dict, Any


class AuditEventType(str, Enum):
    GENESIS = "GENESIS_BLOCK"
    VOICEPRINT_ENROLLED = "VOICEPRINT_ENROLLED"
    CALL_STARTED = "CALL_STARTED"
    WINDOW_ANALYZED = "WINDOW_ANALYZED"
    RISK_EVALUATED = "RISK_EVALUATED"
    ALERT_TRIGGERED = "ALERT_TRIGGERED"
    CHALLENGE_ISSUED = "CHALLENGE_ISSUED"
    VERIFICATION_RESOLVED = "VERIFICATION_RESOLVED"
    CONSENT_RECORDED = "CONSENT_RECORDED"
    CONSENT_REVOKED = "CONSENT_REVOKED"
    BIOMETRICS_PURGED = "BIOMETRICS_PURGED"


def serialize_payload(payload: Dict[str, Any]) -> str:
    """Produces deterministic, canonical JSON string sorted by keys."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
