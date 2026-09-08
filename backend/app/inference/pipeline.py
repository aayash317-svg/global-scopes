"""
Unified Inference Pipeline
Orchestrates Layer 1 multi-signal voice authenticity analysis:
1. Voice Activity Detection (VAD) gate
2. Deepfake detection (Acoustic, Spectral, Prosody signals)
3. Speaker consistency verification against enrolled voiceprint
4. Physical replay attack analysis
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from backend.app.audio.vad import VoiceActivityDetector
from backend.app.inference.deepfake_detector import (
    BaseDeepfakeDetector,
    AASISTDetectorAdapter,
    DeepfakePredictionResult,
)
from backend.app.inference.speaker_consistency import (
    BaseSpeakerConsistency,
    ECAPASpeakerAdapter,
    SpeakerConsistencyResult,
)
from backend.app.inference.replay_detector import (
    BaseReplayDetector,
    AcousticReplayDetector,
    ReplayDetectionResult,
)


@dataclass
class UnifiedInferenceResult:
    # Status
    has_speech: bool
    insufficient_speech: bool
    speech_ratio: float
    
    # Layer 1 - Signal 1: Acoustic
    acoustic_score: float
    
    # Layer 1 - Signal 2: Spectral
    spectral_score: float
    
    # Layer 1 - Signal 3: Prosody
    prosody_score: float
    
    # Layer 1 - Signal 4: Cross-session Speaker Consistency
    speaker_similarity: Optional[float]
    speaker_consistent: Optional[bool]
    
    # Attack Probabilities
    spoof_probability: float
    is_spoof: bool
    replay_probability: float
    is_replay: bool
    
    # Models & Provenance
    deepfake_model: str
    speaker_model: str
    is_mock: bool
    details: Dict[str, Any] = field(default_factory=dict)


class UnifiedInferencePipeline:
    """
    Unified real-time inference pipeline combining VAD gating and Layer 1 detectors.
    Ensures modularity: detectors are injected interfaces rather than tightly coupled.
    """

    def __init__(
        self,
        deepfake_detector: Optional[BaseDeepfakeDetector] = None,
        speaker_verifier: Optional[BaseSpeakerConsistency] = None,
        replay_detector: Optional[BaseReplayDetector] = None,
        vad_detector: Optional[VoiceActivityDetector] = None,
    ):
        self.deepfake_detector = deepfake_detector or AASISTDetectorAdapter()
        self.speaker_verifier = speaker_verifier or ECAPASpeakerAdapter()
        self.replay_detector = replay_detector or AcousticReplayDetector()
        self.vad_detector = vad_detector or VoiceActivityDetector()

    def process_window(
        self,
        audio_window: np.ndarray,
        enrolled_embedding: Optional[np.ndarray] = None,
        sr: int = 16000,
    ) -> UnifiedInferenceResult:
        """
        Analyzes a single 3-4s audio window.
        Executes VAD gate -> Deepfake inference -> Speaker consistency -> Replay detection.
        """
        # 1. Voice Activity Detection Gate
        vad_info = self.vad_detector.detect_speech(audio_window, sr)
        if not vad_info["has_speech"]:
            # Insufficient speech state (Limitation 8 mitigation)
            return UnifiedInferenceResult(
                has_speech=False,
                insufficient_speech=True,
                speech_ratio=vad_info["speech_ratio"],
                acoustic_score=0.0,
                spectral_score=0.0,
                prosody_score=0.0,
                speaker_similarity=None,
                speaker_consistent=None,
                spoof_probability=0.0,
                is_spoof=False,
                replay_probability=0.0,
                is_replay=False,
                deepfake_model=getattr(self.deepfake_detector, "model_name", "AASIST"),
                speaker_model="ECAPA",
                is_mock=getattr(self.deepfake_detector, "is_mock", True),
                details={"vad_info": vad_info, "note": "Gated by VAD: insufficient vocal content"},
            )

        # 2. Deepfake Detection (Acoustic, Spectral, Prosody signals)
        df_res: DeepfakePredictionResult = self.deepfake_detector.predict(audio_window, sr)

        # 3. Speaker Consistency Verification (Signal 4)
        speaker_sim = None
        speaker_consistent = None
        if enrolled_embedding is not None:
            spk_res: SpeakerConsistencyResult = self.speaker_verifier.verify(
                current_audio=audio_window,
                enrolled_embedding=enrolled_embedding,
                threshold=0.75,
                sr=sr,
            )
            speaker_sim = spk_res.similarity
            speaker_consistent = spk_res.is_consistent

        # 4. Replay Attack Detection
        replay_res: ReplayDetectionResult = self.replay_detector.predict(audio_window, sr)

        return UnifiedInferenceResult(
            has_speech=True,
            insufficient_speech=False,
            speech_ratio=vad_info["speech_ratio"],
            acoustic_score=df_res.acoustic_score,
            spectral_score=df_res.spectral_score,
            prosody_score=df_res.prosody_score,
            speaker_similarity=speaker_sim,
            speaker_consistent=speaker_consistent,
            spoof_probability=df_res.spoof_probability,
            is_spoof=df_res.is_spoof,
            replay_probability=replay_res.replay_probability,
            is_replay=replay_res.is_replay,
            deepfake_model=df_res.model_name,
            speaker_model="ECAPA",
            is_mock=df_res.is_mock,
            details={
                "deepfake_details": df_res.details,
                "replay_details": replay_res.details,
                "vad_info": vad_info,
            },
        )
