"""
Retention Policy Subsystem
Automates DPDP Act 2023 Storage Limitation by purging expired sessions and tokens.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import CallSession, VerificationRecord
from config.settings import settings


class RetentionPolicyManager:
    """Enforces automated data retention windows."""

    def __init__(self, retention_days: int = None):
        self.retention_days = retention_days or settings.DATA_RETENTION_DAYS

    async def apply_retention_policy(self, db: AsyncSession) -> Dict[str, Any]:
        """Purges sessions and verification records exceeding retention lifespan."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

        # Delete expired sessions (cascades to audio metadata, detections, risks)
        stmt_sessions = delete(CallSession).where(CallSession.created_at < cutoff_date)
        res_sessions = await db.execute(stmt_sessions)
        deleted_sessions_count = res_sessions.rowcount

        # Delete old verification records
        stmt_verif = delete(VerificationRecord).where(VerificationRecord.created_at < cutoff_date)
        res_verif = await db.execute(stmt_verif)
        deleted_verif_count = res_verif.rowcount

        await db.commit()

        return {
            "retention_days": self.retention_days,
            "cutoff_timestamp": cutoff_date.isoformat(),
            "purged_sessions": deleted_sessions_count,
            "purged_verifications": deleted_verif_count,
        }
