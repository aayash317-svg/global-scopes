"""
Verification API Routes
Handles voiceprint enrollment at KYC stage, step-up MFA, dynamic challenges, and callbacks.
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.session import get_db
from backend.app.database.repositories import PrivacyRepository, VerificationRepository
from backend.app.api.schemas import (
    EnrollVoiceprintResponse,
    ChallengeIssueRequest,
    ChallengeIssueResponse,
    ChallengeVerifyRequest,
    ChallengeVerifyResponse,
    MFAIssueRequest,
    MFAIssueResponse,
    MFAVerifyRequest,
    MFAVerifyResponse,
    VerificationCallbackRequest,
    VerificationCallbackResponse,
)
from backend.app.audio.preprocessing import load_audio, validate_audio_input
from backend.app.inference.speaker_consistency import ECAPASpeakerAdapter
from backend.app.privacy.encryption import BiometricEncryptionService
from backend.app.privacy.consent import ConsentManager
from backend.app.privacy.deletion import DataDeletionManager
from backend.app.verification.challenge import ChallengeService
from backend.app.verification.mfa import MFAService
from backend.app.verification.callback import VerificationCallbackHandler
from backend.app.audit.hash_chain import HashChainService
from backend.app.audit.events import AuditEventType

router = APIRouter()
speaker_adapter = ECAPASpeakerAdapter()
crypto_service = BiometricEncryptionService()
challenge_service = ChallengeService()
mfa_service = MFAService()


@router.post("/enroll", response_model=EnrollVoiceprintResponse, summary="Enroll genuine voice sample at KYC stage")
async def enroll_voiceprint(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    consent_granted: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    """
    KYC Voiceprint Enrollment:
    1. Validates explicit DPDP Act 2023 consent
    2. Decodes audio and extracts fixed-length speaker embedding
    3. Discards raw audio immediately in-memory (Data Minimisation)
    4. Encrypts embedding using Fernet symmetric encryption
    5. Saves encrypted voiceprint and logs event to tamper-evident hash chain
    """
    if not consent_granted:
        raise HTTPException(
            status_code=400,
            detail="Explicit user consent is mandatory for biometric enrollment under DPDP Act 2023.",
        )

    try:
        content = await file.read()
        validate_audio_input(content)
        audio_data, sr = load_audio(content, target_sr=16000)

        # Extract fixed-length speaker embedding
        embedding = speaker_adapter.extract_embedding(audio_data, sr)

        # Immediately zero out and discard raw audio buffer
        DataDeletionManager.zeroize_audio_buffer(audio_data)

        # Encrypt embedding at rest
        encrypted_token = crypto_service.encrypt_embedding(embedding)

        # Store consent and encrypted voiceprint in database
        consent_mgr = ConsentManager(db)
        await consent_mgr.record_enrollment_consent(
            user_id=user_id,
            is_granted=True,
            encrypted_voiceprint=encrypted_token,
        )

        # Record tamper-evident audit block
        hash_svc = HashChainService(db)
        audit_block = await hash_svc.record_event(
            event_type=AuditEventType.VOICEPRINT_ENROLLED.value,
            payload={
                "user_id": user_id,
                "embedding_dimension": len(embedding),
                "encryption_scheme": "Fernet-AES128-CBC-HMAC-SHA256",
                "dpdp_notice": "DPDP-2023-V1",
            },
        )

        return EnrollVoiceprintResponse(
            user_id=user_id,
            enrolled=True,
            message="Voice sample enrolled successfully. Raw audio discarded; biometric encrypted.",
            embedding_dimension=len(embedding),
            audit_event_hash=audit_block.event_hash,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/challenge", response_model=ChallengeIssueResponse, summary="Generate dynamic voice challenge")
async def issue_challenge(
    request: ChallengeIssueRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generates unpredictable dynamic phrase/digits to defeat pre-recorded spoof attacks."""
    prompt = challenge_service.generate_challenge(
        session_id=request.session_id,
        user_id=request.user_id,
        challenge_type=request.challenge_type,
    )

    # Persist verification record
    verif_repo = VerificationRepository(db)
    await verif_repo.create(
        session_id=request.session_id,
        user_id=request.user_id,
        challenge_token=prompt.token,
        verification_type=request.challenge_type,
    )

    # Record audit event
    hash_svc = HashChainService(db)
    await hash_svc.record_event(
        event_type=AuditEventType.CHALLENGE_ISSUED.value,
        payload={"session_id": request.session_id, "user_id": request.user_id, "type": request.challenge_type},
        session_id=request.session_id,
    )

    return ChallengeIssueResponse(
        challenge_id=prompt.challenge_id,
        token=prompt.token,
        challenge_type=prompt.challenge_type,
        prompt_text=prompt.prompt_text,
        expires_in_sec=int(prompt.expires_at - prompt.created_at),
        expected_response=prompt.expected_response,
    )


@router.post("/challenge/verify", response_model=ChallengeVerifyResponse, summary="Verify claimant spoken challenge")
async def verify_challenge_response(
    request: ChallengeVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Verifies claimant spoken response against active challenge prompt."""
    result = challenge_service.verify_challenge(request.token, request.response_text)
    
    verif_repo = VerificationRepository(db)
    status_str = "PASSED" if result["valid"] else ("EXPIRED" if result.get("expired") else "FAILED")
    await verif_repo.update_status(request.token, status_str)

    # Log to audit chain
    hash_svc = HashChainService(db)
    await hash_svc.record_event(
        event_type=AuditEventType.VERIFICATION_RESOLVED.value,
        payload={"token": request.token[:8] + "...", "status": status_str, "valid": result["valid"]},
    )

    return ChallengeVerifyResponse(valid=result["valid"], message=result["reason"])


@router.post("/mfa/dispatch", response_model=MFAIssueResponse, summary="Dispatch step-up OTP alert")
async def dispatch_mfa_otp(request: MFAIssueRequest, db: AsyncSession = Depends(get_db)):
    """Dispatches secondary OTP authentication alert via SMS/Email."""
    otp_data = mfa_service.generate_and_dispatch_otp(
        user_id=request.user_id,
        channel=request.channel,
        recipient=request.recipient,
    )

    verif_repo = VerificationRepository(db)
    await verif_repo.create(
        session_id=request.session_id,
        user_id=request.user_id,
        challenge_token=otp_data["token"],
        verification_type="OTP",
    )

    return MFAIssueResponse(
        token=otp_data["token"],
        channel=otp_data["channel"],
        status=otp_data["status"],
        expires_in_sec=otp_data["expires_in_sec"],
        demo_code=otp_data["demo_code"],
    )


@router.post("/mfa/verify", response_model=MFAVerifyResponse, summary="Verify claimant submitted OTP")
async def verify_mfa_otp(request: MFAVerifyRequest, db: AsyncSession = Depends(get_db)):
    """Verifies claimant OTP submission."""
    res = mfa_service.verify_otp(request.token, request.code)
    
    status_str = "PASSED" if res["valid"] else "FAILED"
    verif_repo = VerificationRepository(db)
    await verif_repo.update_status(request.token, status_str)

    return MFAVerifyResponse(valid=res["valid"], reason=res["reason"])


@router.post("/callback", response_model=VerificationCallbackResponse, summary="Handle out-of-band verification callback")
async def verification_callback(request: VerificationCallbackRequest, db: AsyncSession = Depends(get_db)):
    """Receives callback from external IVR, telecom system, or supervisor console."""
    handler = VerificationCallbackHandler(db)
    result = await handler.handle_callback(
        challenge_token=request.challenge_token,
        verification_status=request.verification_status,
        reviewer_notes=request.reviewer_notes,
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))

    return VerificationCallbackResponse(
        success=True,
        session_id=result["session_id"],
        verification_status=result["verification_status"],
        session_status=result["session_status"],
    )
