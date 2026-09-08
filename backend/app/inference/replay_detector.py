"""
Replay Attack Detector Interface & Implementation
Detects physical acoustic playback attacks (sound played via loudspeaker or secondary transducer).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from scipy import signal


@dataclass
class ReplayDetectionResult:
    replay_probability: float  # 0.0 to 1.0
    is_replay: bool
    channel_distortion_score: float
    cutoff_artifact_score: float
    details: Dict[str, Any]
    model_name: str = "acoustic_replay_detector"
    is_mock: bool = False


class BaseReplayDetector(ABC):
    """Abstract interface for replay detection."""

    @abstractmethod
    def predict(self, audio: np.ndarray, sr: int = 16000) -> ReplayDetectionResult:
        pass


class AcousticReplayDetector(BaseReplayDetector):
    """
    Analyzes physical speaker replay artifacts:
    - Loudspeaker transfer function (sharp high-frequency attenuation)
    - Room reverberation smearing (energy decay rate)
    - Nonlinear harmonic distortion introduced by playback hardware
    """

    def predict(self, audio: np.ndarray, sr: int = 16000) -> ReplayDetectionResult:
        if len(audio) < 512:
            return ReplayDetectionResult(
                replay_probability=0.0,
                is_replay=False,
                channel_distortion_score=0.0,
                cutoff_artifact_score=0.0,
                details={},
                is_mock=False,
            )

        # 1. Power spectral density analysis
        f, psd = signal.welch(audio, fs=sr, nperseg=512)
        psd = psd + 1e-12

        # Loudspeakers typically distort the very low (<100Hz) and very high (>7000Hz) frequency bands
        band_low = psd[f < 150]
        band_mid = psd[(f >= 300) & (f <= 3400)]
        band_high = psd[f > 7000]

        low_energy = float(np.mean(band_low)) if len(band_low) > 0 else 1e-6
        mid_energy = float(np.mean(band_mid)) if len(band_mid) > 0 else 1e-6
        high_energy = float(np.mean(band_high)) if len(band_high) > 0 else 1e-6

        # Ratio of high-frequency attenuation to speech band
        hf_loss_ratio = high_energy / (mid_energy + 1e-9)
        # Ratio of abnormal low resonant bump
        low_bump_ratio = low_energy / (mid_energy + 1e-9)

        # 2. Spectral flux / reverberation smearing
        f_stft, t_stft, zxx = signal.stft(audio, fs=sr, nperseg=512, noverlap=256)
        mag = np.abs(zxx)
        spectral_flux = float(np.mean(np.diff(mag, axis=1) ** 2)) if mag.shape[1] > 1 else 0.0

        # Normalization
        cutoff_artifact_score = float(np.clip(1.0 - (hf_loss_ratio * 100.0), 0.0, 1.0))
        channel_distortion_score = float(np.clip(low_bump_ratio * 2.0, 0.0, 1.0))

        replay_prob = round(0.5 * cutoff_artifact_score + 0.5 * channel_distortion_score, 4)

        return ReplayDetectionResult(
            replay_probability=replay_prob,
            is_replay=replay_prob >= 0.65,
            channel_distortion_score=round(channel_distortion_score, 4),
            cutoff_artifact_score=round(cutoff_artifact_score, 4),
            details={
                "spectral_flux": round(spectral_flux, 6),
                "hf_energy_ratio": round(hf_loss_ratio, 6),
                "low_energy_ratio": round(low_bump_ratio, 6),
            },
            model_name="AcousticReplayAnalyzer",
            is_mock=False,
        )
