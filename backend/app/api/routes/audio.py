"""
Audio Processing API Routes
Thin route controllers delegating to audio subsystems.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Depends
import numpy as np

from backend.app.api.schemas import AudioValidationResponse, TelephonySimulationResponse
from backend.app.audio.preprocessing import load_audio, validate_audio_input
from backend.app.audio.quality import AudioQualityAnalyzer
from backend.app.audio.telephony import TelephonyProcessor
from backend.app.audio.windowing import AudioWindowSegmenter
from backend.app.inference.pipeline import UnifiedInferencePipeline
from backend.app.risk_engine.scorer import RiskScorer
from backend.app.risk_engine.context import RiskContext

router = APIRouter()
quality_analyzer = AudioQualityAnalyzer()
telephony_proc = TelephonyProcessor()
pipeline = UnifiedInferencePipeline()
scorer = RiskScorer()


@router.post("/validate", response_model=AudioValidationResponse, summary="Validate audio quality and suitability")
async def validate_audio(file: UploadFile = File(...)):
    """Validates uploaded audio payload and returns technical SNR/quality metrics."""
    try:
        content = await file.read()
        validate_audio_input(content)
        audio_data, sr = load_audio(content, target_sr=16000)
        q = quality_analyzer.analyze(audio_data, sr)

        return AudioValidationResponse(
            valid=True,
            size_bytes=len(content),
            duration_sec=q["duration_sec"],
            sample_rate=sr,
            snr_db=q["snr_db"],
            clipping_ratio=q["clipping_ratio"],
            quality_rating=q["quality_rating"],
            is_usable=q["is_usable"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/telephony-simulate", response_model=TelephonySimulationResponse, summary="Emulate PSTN/cellular channel degradation")
async def simulate_telephony(file: UploadFile = File(...), codec: str = Form("a_law")):
    """Simulates 8kHz G.711 compression and bandpass telephony degradation."""
    try:
        content = await file.read()
        audio_data, sr = load_audio(content, target_sr=16000)
        degraded = telephony_proc.simulate_pstn_channel(audio_data, sr, codec=codec)

        return TelephonySimulationResponse(
            original_sample_rate=sr,
            telephony_sample_rate=8000,
            codec=codec,
            filtered_samples_count=len(degraded),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.websocket("/stream")
async def audio_stream_websocket(websocket: WebSocket):
    """
    Near-real-time streaming WebSocket endpoint.
    Accepts raw PCM or chunked bytes, runs 3.5s window analysis, and emits dynamic risk alerts.
    """
    await websocket.accept()
    segmenter = AudioWindowSegmenter(window_seconds=3.5, overlap_ratio=0.5)
    audio_accumulator = bytearray()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_accumulator.extend(data)

            # Check if we have at least 1.0 second of 16-bit 16kHz audio (32,000 bytes)
            if len(audio_accumulator) >= 32000:
                # Convert buffer to float32
                samples = np.frombuffer(bytes(audio_accumulator), dtype=np.int16).astype(np.float32) / 32768.0
                windows = segmenter.slice_windows(samples, sr=16000)

                for w in windows:
                    inf = pipeline.process_window(w.samples, sr=16000)
                    risk = scorer.assess_risk(inf, RiskContext(scenario="routine_support"))

                    await websocket.send_json({
                        "window_index": w.index,
                        "start_sec": w.start_sec,
                        "end_sec": w.end_sec,
                        "has_speech": inf.has_speech,
                        "spoof_probability": inf.spoof_probability,
                        "risk_score": risk.risk_score,
                        "risk_tier": risk.risk_tier,
                        "floor_applied": risk.safety_floor_applied,
                        "reasons": risk.reasons,
                    })

                # Retain last overlapping portion
                overlap_bytes = int(len(audio_accumulator) * 0.5)
                audio_accumulator = bytearray(audio_accumulator[-overlap_bytes:])
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
