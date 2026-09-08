"""
API Integration Tests
Validates all REST endpoints via httpx AsyncClient.
"""

import pytest


@pytest.mark.asyncio
async def test_api_compliance_endpoint(client):
    response = await client.get("/compliance")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Compliant"
    assert "DPDP Act 2023" in data["act"]
    assert data["raw_audio_persisted"] is False


@pytest.mark.asyncio
async def test_api_enrollment_endpoint(client, synthetic_wav_bytes):
    response = await client.post(
        "/enroll",
        data={"user_id": "cxo_executive", "consent_granted": "true"},
        files={"file": ("sample.wav", synthetic_wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["enrolled"] is True
    assert data["user_id"] == "cxo_executive"
    assert "audit_event_hash" in data


@pytest.mark.asyncio
async def test_api_analyze_pipeline(client, synthetic_wav_bytes):
    response = await client.post(
        "/analyze",
        data={
            "caller_id": "caller_test",
            "scenario": "routine_support",
            "call_origin": "VoIP",
            "transaction_value": "5000.0",
        },
        files={"file": ("chunk.wav", synthetic_wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_tier" in data
    assert "signals" in data
    assert "audit_event_hash" in data


@pytest.mark.asyncio
async def test_api_risk_assess_and_thresholds(client):
    # Get thresholds
    th_res = await client.get("/api/v1/risk/thresholds")
    assert th_res.status_code == 200
    assert len(th_res.json()) >= 3

    # Calculate risk with high single signal to trigger safety floor
    assess_res = await client.post(
        "/api/v1/risk/assess",
        json={
            "caller_id": "user_risk_test",
            "scenario": "routine_support",
            "acoustic_score": 0.99,
            "spectral_score": 0.10,
            "prosody_score": 0.10,
            "transaction_value": 0.0,
        },
    )
    assert assess_res.status_code == 200
    data = assess_res.json()
    assert data["risk_tier"] in ("HIGH", "CRITICAL")
    assert data["safety_floor_applied"] is True


@pytest.mark.asyncio
async def test_api_verification_flow(client):
    # 1. Issue challenge
    chal_res = await client.post(
        "/api/v1/verification/challenge",
        json={"session_id": "session_verif_01", "user_id": "test_user"},
    )
    assert chal_res.status_code == 200
    c_data = chal_res.json()
    token = c_data["token"]
    expected = c_data["expected_response"]

    # 2. Verify challenge
    verif_res = await client.post(
        "/api/v1/verification/challenge/verify",
        json={"token": token, "response_text": expected},
    )
    assert verif_res.status_code == 200
    assert verif_res.json()["valid"] is True


@pytest.mark.asyncio
async def test_api_audit_verify(client):
    res = await client.get("/audit/verify")
    assert res.status_code == 200
    data = res.json()
    assert "is_valid" in data
