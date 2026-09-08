"""
Verification Subsystem Tests
Validates dynamic challenge generation, response matching, and MFA OTP workflows.
"""

import pytest

from backend.app.verification.challenge import ChallengeService
from backend.app.verification.mfa import MFAService


def test_challenge_generation_and_verification():
    service = ChallengeService(default_ttl_sec=60)
    prompt = service.generate_challenge(
        session_id="session_001",
        user_id="user_test",
        challenge_type="numeric_sequence",
    )

    assert len(prompt.token) > 10
    assert prompt.expected_response is not None

    # Verify correct response
    res_correct = service.verify_challenge(prompt.token, prompt.expected_response)
    assert res_correct["valid"] is True

    # Re-verifying same token should fail (single-use)
    res_reuse = service.verify_challenge(prompt.token, prompt.expected_response)
    assert res_reuse["valid"] is False


def test_mfa_otp_lifecycle():
    mfa = MFAService(otp_ttl_sec=60, max_attempts=3)
    otp_data = mfa.generate_and_dispatch_otp(user_id="user_mfa")

    token = otp_data["token"]
    code = otp_data["demo_code"]

    # Wrong code attempt
    res_wrong = mfa.verify_otp(token, "000000")
    assert res_wrong["valid"] is False

    # Correct code
    res_right = mfa.verify_otp(token, code)
    assert res_right["valid"] is True


def test_supervisor_escalation_alert():
    mfa = MFAService()
    alert = mfa.dispatch_supervisor_alert(
        session_id="session_flagged",
        caller_id="attacker_1",
        risk_score=94.5,
        reasons=["High acoustic jitter", "Voiceprint mismatch"],
    )
    assert alert["status"] == "ESCALATED"
    assert len(mfa.sent_alerts) > 0
