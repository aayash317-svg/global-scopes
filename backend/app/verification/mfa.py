"""
Multi-Factor Authentication (MFA) & Step-Up Alerting Subsystem
Handles OTP dispatch, multi-channel notification simulation, and supervisor escalation.
"""

import hmac
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("voice_integrity.mfa")


@dataclass
class OTPRecord:
    token: str
    user_id: str
    code_hash: str
    salt: str
    attempts: int
    max_attempts: int
    expires_at: float


class MFAService:
    """Manages secondary authentication and out-of-band alerting."""

    def __init__(self, otp_ttl_sec: int = 300, max_attempts: int = 3):
        self.otp_ttl_sec = otp_ttl_sec
        self.max_attempts = max_attempts
        self._otps: Dict[str, OTPRecord] = {}
        self.sent_alerts: List[Dict[str, str]] = []

    def _hash_code(self, code: str, salt: str) -> str:
        return hashlib.sha256((code + salt).encode("utf-8")).hexdigest()

    def generate_and_dispatch_otp(
        self,
        user_id: str,
        channel: str = "SMS",
        recipient: str = "+91-XXXXXXXXXX",
    ) -> Dict[str, str]:
        """Generates a secure 6-digit OTP and simulates multi-channel dispatch."""
        code = f"{secrets.randbelow(900000) + 100000}"  # 6-digit numeric string
        salt = secrets.token_hex(8)
        token = secrets.token_urlsafe(16)
        expires_at = time.time() + self.otp_ttl_sec

        self._otps[token] = OTPRecord(
            token=token,
            user_id=user_id,
            code_hash=self._hash_code(code, salt),
            salt=salt,
            attempts=0,
            max_attempts=self.max_attempts,
            expires_at=expires_at,
        )

        # Dispatch simulation
        msg = f"[Voice Integrity Verification] High-risk voice anomaly detected. Your one-time authorization code is: {code}"
        self.sent_alerts.append({
            "channel": channel,
            "recipient": recipient,
            "message": msg,
            "timestamp": time.time(),
        })
        logger.info(f"Dispatched {channel} step-up alert to {recipient} (token: {token})")

        return {
            "token": token,
            "channel": channel,
            "status": "DISPATCHED",
            "expires_in_sec": self.otp_ttl_sec,
            # For testing/demo mode convenience:
            "demo_code": code,
        }

    def verify_otp(self, token: str, code: str) -> Dict[str, bool]:
        """Validates claimant OTP submission against stored cryptographic hash."""
        record = self._otps.get(token)
        if not record:
            return {"valid": False, "reason": "Invalid or missing OTP token."}

        if time.time() > record.expires_at:
            del self._otps[token]
            return {"valid": False, "reason": "OTP code has expired."}

        if record.attempts >= record.max_attempts:
            del self._otps[token]
            return {"valid": False, "reason": "Maximum verification attempts exceeded."}

        record.attempts += 1
        test_hash = self._hash_code(code.strip(), record.salt)

        if hmac.compare_digest(test_hash, record.code_hash):
            del self._otps[token]
            return {"valid": True, "reason": "OTP verified successfully."}

        remaining = record.max_attempts - record.attempts
        return {
            "valid": False,
            "reason": f"Incorrect OTP code. {remaining} attempt(s) remaining.",
        }

    def dispatch_supervisor_alert(self, session_id: str, caller_id: str, risk_score: float, reasons: List[str]) -> Dict[str, str]:
        """Dispatches immediate high-priority escalation alert to supervisor dashboard."""
        alert = {
            "type": "SUPERVISOR_TAKEOVER_ALERT",
            "session_id": session_id,
            "caller_id": caller_id,
            "risk_score": f"{risk_score:.1f}",
            "reasons": "; ".join(reasons),
            "timestamp": time.time(),
        }
        self.sent_alerts.append(alert)
        logger.warning(f"SUPERVISOR ESCALATION: Session {session_id} flagged with risk score {risk_score}")
        return {"status": "ESCALATED", "session_id": session_id}
