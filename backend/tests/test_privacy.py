"""
Privacy and DPDP Act 2023 Compliance Tests
Validates Fernet encryption/decryption, data minimisation, memory zeroization, and retention.
"""

import pytest
import numpy as np

from backend.app.privacy.encryption import BiometricEncryptionService
from backend.app.privacy.deletion import DataDeletionManager
from backend.app.privacy.consent import ConsentManager
from backend.app.privacy.retention import RetentionPolicyManager
from backend.app.database.repositories import SessionRepository


def test_biometric_fernet_encryption_roundtrip():
    crypto = BiometricEncryptionService()
    embedding = np.random.randn(78).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    encrypted_str = crypto.encrypt_embedding(embedding)
    assert isinstance(encrypted_str, str)
    assert len(encrypted_str) > 0

    decrypted = crypto.decrypt_embedding(encrypted_str)
    assert np.allclose(embedding, decrypted, atol=1e-5)


def test_memory_zeroization():
    audio_buffer = np.ones((16000,), dtype=np.float32)
    DataDeletionManager.zeroize_audio_buffer(audio_buffer)
    assert np.sum(audio_buffer) == 0.0


@pytest.mark.asyncio
async def test_consent_and_erasure(db_session):
    consent_mgr = ConsentManager(db_session)
    user_id = "test_kyc_user"

    # 1. Record consent
    rec = await consent_mgr.record_enrollment_consent(
        user_id=user_id,
        is_granted=True,
        encrypted_voiceprint="fernet_ciphertext_dummy",
    )
    assert rec["consent_granted"] is True

    # Verify active consent
    is_valid = await consent_mgr.verify_consent(user_id)
    assert is_valid is True

    # 2. Right to be forgotten (Erasure)
    purge_res = await DataDeletionManager.purge_user_data(user_id, db_session)
    assert purge_res["biometric_voiceprint_erased"] is True

    # Consent verify should now be False
    is_valid_after = await consent_mgr.verify_consent(user_id)
    assert is_valid_after is False


@pytest.mark.asyncio
async def test_retention_policy_sweep(db_session):
    session_repo = SessionRepository(db_session)
    await session_repo.create("caller_test_retention")

    ret_mgr = RetentionPolicyManager(retention_days=30)
    sweep_res = await ret_mgr.apply_retention_policy(db_session)
    assert "purged_sessions" in sweep_res
