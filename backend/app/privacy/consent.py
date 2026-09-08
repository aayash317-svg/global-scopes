"""
DPDP Act 2023 Consent Management Subsystem
Enforces explicit consent acquisition, notice delivery, and revocation
for voice biometric enrollment and processing.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.repositories import PrivacyRepository


class ConsentManager:
    """Manages DPDP Act 2023 Section 6 consent lifecycle."""

    DEFAULT_PURPOSE = "fraud_prevention_voice_verification"
    KYC_NOTICE_TEXT = (
        "In accordance with the Digital Personal Data Protection Act 2023, your voice sample "
        "will be converted into an encrypted mathematical biometric embedding strictly for "
        "fraud detection and identity verification during sensitive transactions. Raw audio is "
        "discarded immediately in-memory. You may revoke consent at any time."
    )

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = PrivacyRepository(db)

    async def record_enrollment_consent(
        self,
        user_id: str,
        is_granted: bool = True,
        encrypted_voiceprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records explicit consent and associates encrypted voiceprint."""
        if not is_granted:
            raise ValueError("Explicit consent is required for voice biometric enrollment under DPDP Act 2023.")

        record = await self.repo.upsert_consent(
            user_id=user_id,
            is_granted=True,
            encrypted_voiceprint=encrypted_voiceprint,
            purpose=self.DEFAULT_PURPOSE,
        )

        return {
            "user_id": user_id,
            "consent_granted": True,
            "purpose": self.DEFAULT_PURPOSE,
            "notice_version": "DPDP-2023-V1",
            "timestamp": record.granted_at.isoformat() if record.granted_at else datetime.now(timezone.utc).isoformat(),
        }

    async def verify_consent(self, user_id: str) -> bool:
        """Verifies whether active, unrevoked consent exists for user."""
        record = await self.repo.get_consent(user_id)
        if not record or not record.is_granted:
            return False
        return True

    async def revoke_consent(self, user_id: str) -> Dict[str, Any]:
        """Revokes consent and flags voiceprint for deletion."""
        record = await self.repo.upsert_consent(
            user_id=user_id,
            is_granted=False,
            encrypted_voiceprint=None,
        )
        return {
            "user_id": user_id,
            "consent_granted": False,
            "status": "REVOKED",
            "revoked_at": datetime.now(timezone.utc).isoformat(),
        }
