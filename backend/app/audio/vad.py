"""
Voice Activity Detection (VAD) Subsystem
Gates non-speech, hold music, dial tones, and silence before model inference.
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class VoiceActivityDetector:
    """
    Lightweight energy and zero-crossing rate (ZCR) based Voice Activity Detector
    with adaptive thresholding and hangover smoothing.
    """

    def __init__(
        self,
        frame_duration_ms: float = 25.0,
        energy_threshold_multiplier: float = 2.5,
        min_speech_duration_sec: float = 0.5,
        hangover_frames: int = 5,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.min_speech_duration_sec = min_speech_duration_sec
        self.hangover_frames = hangover_frames

    def detect_speech(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Detects active speech regions in audio array.
        Returns speech ratio, total speech duration, and detected segment boundaries.
        """
        if len(audio) == 0:
            return {
                "has_speech": False,
                "speech_ratio": 0.0,
                "speech_duration_sec": 0.0,
                "speech_segments": [],
                "speech_frames_count": 0,
                "total_frames_count": 0,
            }

        frame_len = int(sr * (self.frame_duration_ms / 1000.0))
        if frame_len <= 0:
            frame_len = 400

        n_frames = len(audio) // frame_len
        if n_frames == 0:
            return {
                "has_speech": False,
                "speech_ratio": 0.0,
                "speech_duration_sec": 0.0,
                "speech_segments": [],
                "speech_frames_count": 0,
                "total_frames_count": 0,
            }

        # Reshape into frames
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)

        # Frame energy (Root Mean Square)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1) / (2.0 * frame_len)

        # Adaptive noise floor estimation: 15th percentile of RMS energy
        # Capped so loud continuous vocal tones are not mistakenly treated as noise floor
        noise_floor = min(float(np.percentile(frame_rms, 15)), 0.02)
        energy_threshold = max(noise_floor * self.energy_threshold_multiplier, 0.01)

        # Raw frame classification: high energy and plausible speech ZCR (< 0.45)
        is_speech_frame = (frame_rms > energy_threshold) & (zero_crossings < 0.45)

        # Hangover smoothing: maintain speech state across brief pauses (e.g. stop consonants)
        smoothed = np.copy(is_speech_frame)
        hangover = 0
        for i in range(n_frames):
            if is_speech_frame[i]:
                hangover = self.hangover_frames
                smoothed[i] = True
            elif hangover > 0:
                smoothed[i] = True
                hangover -= 1

        speech_frames_count = int(np.sum(smoothed))
        speech_ratio = float(speech_frames_count / n_frames)
        speech_duration_sec = speech_frames_count * (frame_len / sr)

        # Extract continuous speech segments
        speech_segments: List[Tuple[float, float]] = []
        in_segment = False
        start_frame = 0

        for i, val in enumerate(smoothed):
            if val and not in_segment:
                in_segment = True
                start_frame = i
            elif not val and in_segment:
                in_segment = False
                speech_segments.append(
                    (round(start_frame * (frame_len / sr), 3), round(i * (frame_len / sr), 3))
                )

        if in_segment:
            speech_segments.append(
                (round(start_frame * (frame_len / sr), 3), round(n_frames * (frame_len / sr), 3))
            )

        has_speech = speech_duration_sec >= self.min_speech_duration_sec

        return {
            "has_speech": bool(has_speech),
            "speech_ratio": round(speech_ratio, 4),
            "speech_duration_sec": round(speech_duration_sec, 3),
            "speech_segments": speech_segments,
            "speech_frames_count": speech_frames_count,
            "total_frames_count": n_frames,
        }

    def filter_speech_only(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extracts and concatenates only active speech regions."""
        detection = self.detect_speech(audio, sr)
        if not detection["speech_segments"]:
            return np.array([], dtype=np.float32)

        selected_chunks = []
        for start_sec, end_sec in detection["speech_segments"]:
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            selected_chunks.append(audio[start_idx:end_idx])

        if selected_chunks:
            return np.concatenate(selected_chunks).astype(np.float32)
        return np.array([], dtype=np.float32)
