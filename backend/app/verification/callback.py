"""
Verification Callback Handler
Processes asynchronous out-of-band step-up verification responses
from telecom IVR, supervisor consoles, and customer portals.
"""

from typing import Dict, Any, Optional
import time
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.repositories import VerificationRepository, SessionRepository

logger = logging.getLogger("voice_integrity.verification_callback")


class VerificationCallbackHandler:
    """Processes verification callbacks and updates persistent session status."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.verif_repo = VerificationRepository(db)
        self.session_repo = SessionRepository(db)

    async def handle_callback(
        self,
        challenge_token: str,
        verification_status: str,  # "PASSED", "FAILED", "EXPIRED"
        reviewer_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handles callback resolution for an active verification challenge."""
        record = await self.verif_repo.get_by_token(challenge_token)
        if not record:
            return {
                "success": False,
                "message": f"Verification record with token {challenge_token} not found.",
            }

        # Update verification record
        updated_verif = await self.verif_repo.update_status(challenge_token, verification_status)

        # Update parent call session status
        if verification_status == "PASSED":
            session_status = "ACTIVE"
        else:
            session_status = "FLAGGED"

        await self.session_repo.update_status(record.session_id, session_status)

        logger.info(
            f"Verification callback for session {record.session_id}: {verification_status} (Notes: {reviewer_notes})"
        )

        return {
            "success": True,
            "session_id": record.session_id,
            "verification_status": verification_status,
            "session_status": session_status,
            "completed_at": time.time(),
        }
