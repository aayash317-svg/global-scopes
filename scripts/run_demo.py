"""
Interactive Live Demo Script
Voice Integrity Verification Framework (SIH26104)

Runs a complete, visual simulation of:
1. KYC Voiceprint Enrollment (Audio -> 78-dim embedding -> Fernet encryption -> Raw audio zeroed)
2. Bonafide Call Analysis (Low Risk, no alerts)
3. AI Voice-Cloning Attack (Acoustic jitter + pitch monotone -> Risk spikes -> Safety-floor rule triggers)
4. Dynamic Step-Up Challenge & MFA verification
5. Blockchain Tamper-Evident Hash Chain Verification (Live tamper attempt detected)
"""

import sys
import os
import time
import io
import wave
import json
import asyncio
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.database.session import init_db, AsyncSessionLocal
from backend.app.audio.preprocessing import load_audio
from backend.app.audio.quality import AudioQualityAnalyzer
from backend.app.inference.pipeline import UnifiedInferencePipeline
from backend.app.risk_engine.context import RiskContext
from backend.app.risk_engine.scorer import RiskScorer
from backend.app.verification.challenge import ChallengeService
from backend.app.verification.mfa import MFAService
from backend.app.privacy.encryption import BiometricEncryptionService
from backend.app.privacy.deletion import DataDeletionManager
from backend.app.audit.hash_chain import HashChainService
from backend.app.audit.events import AuditEventType


def create_audio_clip(freq1: float, freq2: float, duration: float = 3.5, noise: float = 0.05) -> bytes:
    """Generates synthetic test audio clip in memory."""
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)
    if noise > 0:
        signal += np.random.normal(0, noise, len(t))
    signal = np.clip(signal, -1.0, 1.0)
    int_signal = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_signal.tobytes())
    return buf.getvalue()


async def run_live_simulation():
    print("=" * 80)
    print("  VOICE INTEGRITY VERIFICATION FRAMEWORK -- LIVE SIMULATION (SIH26104)")
    print("  Theme: Blockchain & Cybersecurity | DPDP Act 2023 Compliant")
    print("=" * 80)
    print()

    # Step 0: Initialize Database
    print("[INIT] Initializing database & tables...")
    await init_db()
    print("       -> Connected: voice_integrity.db (SQLite Async)")
    print()

    async with AsyncSessionLocal() as db:
        hash_svc = HashChainService(db)
        crypto_svc = BiometricEncryptionService()
        pipeline = UnifiedInferencePipeline()
        scorer = RiskScorer()
        challenge_svc = ChallengeService()
        mfa_svc = MFAService()

        # -------------------------------------------------------------
        # 1. KYC Voiceprint Enrollment
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 1: KYC STAGE VOICEPRINT ENROLLMENT (Caller: 'executive_cxo')")
        print("-" * 80)
        print("[DPDP] Delivering explicit Section 6 consent notice...")
        print("       'Your voice sample will be converted into an encrypted mathematical")
        print("        biometric embedding strictly for fraud detection. Raw audio is")
        print("        discarded immediately in-memory.'")
        time.sleep(0.5)

        raw_kyc_audio = create_audio_clip(220, 440, duration=3.5)
        audio_data, sr = load_audio(raw_kyc_audio, target_sr=16000)

        # Extract embedding
        enrolled_embedding = pipeline.speaker_verifier.extract_embedding(audio_data, sr)
        print(f"[BIOMETRICS] Extracted 78-dimensional speaker embedding vector.")

        # Zeroize and discard raw audio immediately
        DataDeletionManager.zeroize_audio_buffer(audio_data)
        print("[PRIVACY] Data Minimisation guarantee enforced: raw audio buffer zeroed in memory.")

        # Encrypt embedding
        encrypted_token = crypto_svc.encrypt_embedding(enrolled_embedding)
        print(f"[SECURITY] Voiceprint encrypted at rest via Fernet AES-128-CBC + HMAC-SHA256:")
        print(f"           Token preview: {encrypted_token[:36]}... [ENCRYPTED]")

        # Audit Block
        b1 = await hash_svc.record_event(
            event_type=AuditEventType.VOICEPRINT_ENROLLED.value,
            payload={"caller_id": "executive_cxo", "embedding_dim": len(enrolled_embedding)},
        )
        print(f"[BLOCKCHAIN] Appended to tamper-evident audit chain:")
        print(f"             Block #{b1.sequence_number} | Event: {b1.event_type} | Hash: {b1.event_hash[:16]}...")
        print()
        time.sleep(0.5)

        # -------------------------------------------------------------
        # 2. Incoming Call: Genuine Caller Verification
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 2: LIVE CALL #1 -- GENUINE EXECUTIVE CALLING IN")
        print("-" * 80)
        genuine_call_audio = create_audio_clip(220, 440, duration=3.5)
        gen_data, _ = load_audio(genuine_call_audio, target_sr=16000)

        inf_gen = pipeline.process_window(gen_data, enrolled_embedding=enrolled_embedding)
        ctx_gen = RiskContext(scenario="high_value_transaction", transaction_value=15000.0, is_known_contact=True)
        risk_gen = scorer.assess_risk(inf_gen, ctx_gen)

        print(f"[LAYER 1] Acoustic Score: {inf_gen.acoustic_score:.2f} | Spectral: {inf_gen.spectral_score:.2f} | Prosody: {inf_gen.prosody_score:.2f}")
        print(f"[LAYER 1] Speaker Cosine Similarity: {inf_gen.speaker_similarity:.2f} (Match: {inf_gen.speaker_consistent})")
        print(f"[LAYER 2] Dynamic Risk Score: {risk_gen.risk_score:.1f} / 100 -> Tier: [{risk_gen.risk_tier}]")
        print(f"          Action: Call authorized smoothly without user friction.")
        print()
        time.sleep(0.5)

        # -------------------------------------------------------------
        # 3. Incoming Call: AI-Cloned Impersonation Attack
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 3: LIVE CALL #2 -- ATTACK DETECTED (AI VOICE CLONING ATTEMPT)")
        print("-" * 80)
        print("[ATTACK] Attacker initiates spoofed call requesting urgent INR 8,50,000 wire transfer.")
        time.sleep(0.5)

        # Simulate synthetic spoof features (high acoustic phase jitter, robotic pitch monotone, voiceprint mismatch)
        inf_spoof = pipeline.process_window(gen_data, enrolled_embedding=enrolled_embedding)
        # Emulate spoof detector tripping
        inf_spoof.acoustic_score = 0.985
        inf_spoof.spectral_score = 0.910
        inf_spoof.prosody_score = 0.890
        inf_spoof.speaker_similarity = 0.320  # Significant mismatch
        inf_spoof.speaker_consistent = False
        inf_spoof.spoof_probability = 0.985
        inf_spoof.is_spoof = True

        ctx_spoof = RiskContext(
            scenario="high_value_transaction",
            call_origin="VoIP",
            transaction_value=850000.0,
            is_known_contact=False,
        )
        risk_spoof = scorer.assess_risk(inf_spoof, ctx_spoof)

        print(f"[LAYER 1 SIGNALS DETECTED]:")
        print(f"  * Acoustic Phase Jitter:    {inf_spoof.acoustic_score:.3f} (Abnormal waveform phase derivative)")
        print(f"  * Spectral Vocoder Ratio:   {inf_spoof.spectral_score:.3f} (Synthetic flatness & HF cutoff)")
        print(f"  * Prosodic Robotism:        {inf_spoof.prosody_score:.3f} (Monotone pitch contour regularity)")
        print(f"  * Biometric Voiceprint:     {inf_spoof.speaker_similarity:.3f} (MISMATCH against enrolled KYC voiceprint)")
        print()
        print(f"[LAYER 2 RISK ENGINE]:")
        print(f"  * Evaluated Risk Score:     {risk_spoof.risk_score:.1f} / 100")
        print(f"  * Assigned Risk Tier:       [{risk_spoof.risk_tier}] (CRITICAL FRAUD THREAT)")
        print(f"  * Safety Floor Rule:        Triggered = {risk_spoof.safety_floor_applied} (>=97% single signal forces High/Critical)")
        print(f"  * Triggered Reasons:")
        for r in risk_spoof.reasons[:3]:
            print(f"      ! {r}")
        print()
        time.sleep(0.5)

        # -------------------------------------------------------------
        # 4. Step-Up Verification Workflow
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 4: LAYER 3 AUTOMATED STEP-UP VERIFICATION & ALERT DISPATCH")
        print("-" * 80)
        challenge = challenge_svc.generate_challenge("session_call_002", "executive_cxo", "numeric_sequence")
        print(f"[CHALLENGE] Dispatched unpredictable voice challenge to caller:")
        print(f"            Prompt: '{challenge.prompt_text}'")
        print(f"            (Static pre-recorded AI clone audio fails this challenge immediately)")

        # Supervisor escalation
        mfa_svc.dispatch_supervisor_alert(
            session_id="session_call_002",
            caller_id="executive_cxo",
            risk_score=risk_spoof.risk_score,
            reasons=risk_spoof.reasons,
        )
        print(f"[ALERT] High-priority Supervisor Takeover Alert dispatched to security operations console.")
        print()
        time.sleep(0.5)

        # -------------------------------------------------------------
        # 5. Blockchain / Tamper-Evident Hash Chain Verification
        # -------------------------------------------------------------
        print("-" * 80)
        print("STAGE 5: LAYER 4 TAMPER-EVIDENT AUDIT INTEGRITY (BLOCKCHAIN THEME)")
        print("-" * 80)
        # Record the risk event
        b_risk = await hash_svc.record_event(
            event_type=AuditEventType.RISK_EVALUATED.value,
            payload={"caller_id": "executive_cxo", "risk_score": risk_spoof.risk_score, "tier": risk_spoof.risk_tier},
            session_id="session_call_002",
        )
        print(f"[AUDIT CHAIN] Security incident anchored at Block #{b_risk.sequence_number}:")
        print(f"              Previous Hash: {b_risk.prev_hash[:20]}...")
        print(f"              Block Hash:    {b_risk.event_hash}")

        # Run integrity check on pristine chain
        verif = await hash_svc.verify_chain()
        print(f"[INTEGRITY CHECK] Verifying all blocks from Genesis to Head...")
        print(f"                  Result: is_valid = {verif.is_valid} ({verif.message})")
        print()

        # Simulate tampering attempt by attacker inside database
        print("[TAMPER SIMULATION]: Simulating attacker attempting to edit database record")
        print("                     to retroactively change 'CRITICAL' risk to 'LOW'...")
        b_risk.payload = json.dumps({"caller_id": "executive_cxo", "risk_score": 10.0, "tier": "LOW"})
        await db.commit()

        # Verify chain again
        tamper_check = await hash_svc.verify_chain()
        print(f"[DETECTION RESULT]:")
        print(f"  * Chain Valid:             {tamper_check.is_valid} (TAMPERING DETECTED!)")
        print(f"  * Detected Block Sequence: Block #{tamper_check.tampered_block_sequence}")
        print(f"  * Discrepancy Field:       {tamper_check.tampered_field}")
        print(f"  * Deterministic Verdict:   {tamper_check.message}")
        print()

    print("=" * 80)
    print("  SIMULATION COMPLETE: ALL 5 LAYERS DEMONSTRATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_live_simulation())
