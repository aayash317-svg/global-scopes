"""
Privacy & Compliance API Routes
Implements DPDP Act 2023 compliance endpoints: consent, deletion, and privacy reporting.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.session import get_db
from backend.app.api.schemas import ConsentRequest, ConsentResponse, ComplianceResponse
from backend.app.privacy.consent import ConsentManager
from backend.app.privacy.deletion import DataDeletionManager
from backend.app.privacy.retention import RetentionPolicyManager
from backend.app.audit.hash_chain import HashChainService
from backend.app.audit.events import AuditEventType
from config.settings import settings

router = APIRouter()


@router.get("/consent/{user_id}", response_model=ConsentResponse, summary="Check user consent status")
async def get_consent_status(user_id: str, db: AsyncSession = Depends(get_db)):
    """Queries active DPDP consent record for given user ID."""
    mgr = ConsentManager(db)
    is_granted = await mgr.verify_consent(user_id)
    record = await mgr.repo.get_consent(user_id)

    if not record:
        raise HTTPException(status_code=404, detail=f"No consent record found for user {user_id}")

    return ConsentResponse(
        user_id=user_id,
        consent_granted=is_granted,
        purpose=record.purpose,
        timestamp=record.granted_at.isoformat() if record.granted_at else "",
    )


@router.post("/consent", response_model=ConsentResponse, summary="Record or update DPDP consent")
async def set_consent(request: ConsentRequest, db: AsyncSession = Depends(get_db)):
    """Records user explicit consent or revocation."""
    mgr = ConsentManager(db)
    if request.is_granted:
        res = await mgr.record_enrollment_consent(request.user_id, is_granted=True)
    else:
        res = await mgr.revoke_consent(request.user_id)

    # Log to audit chain
    hash_svc = HashChainService(db)
    await hash_svc.record_event(
        event_type=AuditEventType.CONSENT_RECORDED.value if request.is_granted else AuditEventType.CONSENT_REVOKED.value,
        payload={"user_id": request.user_id, "is_granted": request.is_granted},
    )

    return ConsentResponse(
        user_id=request.user_id,
        consent_granted=request.is_granted,
        purpose=mgr.DEFAULT_PURPOSE,
        timestamp=res.get("timestamp", ""),
    )


@router.delete("/user/{user_id}", summary="Right to be Forgotten (DPDP Section 12)")
async def delete_user_biometrics(user_id: str, db: AsyncSession = Depends(get_db)):
    """Permanently purges all encrypted biometric voiceprints and revoked profiles."""
    res = await DataDeletionManager.purge_user_data(user_id, db)

    # Log to audit chain
    hash_svc = HashChainService(db)
    await hash_svc.record_event(
        event_type=AuditEventType.BIOMETRICS_PURGED.value,
        payload={"user_id": user_id, "status": "PERMANENTLY_PURGED"},
    )

    return res


@router.get("/compliance", response_model=ComplianceResponse, summary="DPDP Act 2023 compliance mapping")
async def get_compliance_info():
    """Returns technical mapping demonstrating compliance with DPDP Act 2023 principles."""
    return ComplianceResponse(
        act="Digital Personal Data Protection Act 2023 (DPDP Act 2023 & 2025 Rules)",
        status="Compliant",
        principles={
            "Data Minimisation": "Raw audio processed transiently in-memory per 3.5s window and immediately zeroed/discarded; zero raw audio persisted.",
            "Storage Limitation": f"Audio discarded post-inference; metadata subject to automated {settings.DATA_RETENTION_DAYS}-day retention cleanup.",
            "Purpose Limitation": "Biometric embeddings restricted strictly to fraud prevention and identity verification.",
            "Consent Requirement": "Mandatory explicit consent notice delivered and verified prior to KYC voiceprint enrollment.",
            "Security Safeguards": "Fernet AES-128-CBC authenticated encryption for stored voiceprints; SHA-256 tamper-evident hash-chain audit trail.",
            "Right to Erasure": "Full cryptographic purge endpoint allowing immediate, permanent deletion of enrolled voiceprints.",
        },
        retention_days=settings.DATA_RETENTION_DAYS,
        biometric_encryption="Fernet symmetric authenticated encryption (KMS/HSM upgrade ready)",
        raw_audio_persisted=False,
    )


@router.post("/retention/apply", summary="Execute automated data retention cleanup")
async def run_retention_sweep(db: AsyncSession = Depends(get_db)):
    """Triggers retention policy sweep deleting records past retention lifespan."""
    ret_mgr = RetentionPolicyManager()
    res = await ret_mgr.apply_retention_policy(db)
    return res
