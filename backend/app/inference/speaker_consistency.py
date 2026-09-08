"""
Speaker Consistency Verification Subsystem
Implements Layer 1 Signal 4: Cross-session speaker verification using VoxCeleb/ECAPA embedding interfaces.
Compares real-time caller embeddings to enrolled voiceprints via cosine similarity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import signal, fft

from config.settings import settings


@dataclass
class SpeakerConsistencyResult:
    similarity: float  # -1.0 to 1.0 (typically 0.0 to 1.0)
    is_consistent: bool
    threshold: float
    model_name: str
    is_mock: bool = False


class BaseSpeakerConsistency(ABC):
    """Abstract interface for speaker embedding extraction and consistency verification."""

    @abstractmethod
    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extracts fixed-dimension 1D float32 speaker embedding vector."""
        pass

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two normalized speaker embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(round(np.clip(cosine_sim, -1.0, 1.0), 4))

    def verify(
        self,
        current_audio: np.ndarray,
        enrolled_embedding: np.ndarray,
        threshold: float = 0.75,
        sr: int = 16000,
    ) -> SpeakerConsistencyResult:
        """Verifies if current audio window matches enrolled voiceprint."""
        current_emb = self.extract_embedding(current_audio, sr)
        sim = self.compute_similarity(current_emb, enrolled_embedding)
        return SpeakerConsistencyResult(
            similarity=sim,
            is_consistent=sim >= threshold,
            threshold=threshold,
            model_name="ECAPA-TDNN" if not getattr(self, "is_mock", True) else "ECAPA-MFCC-Stub",
            is_mock=getattr(self, "is_mock", True),
        )


class MFCCSpeakerEmbeddingExtractor:
    """
    Classical acoustic feature embedding generator:
    Computes 13 Mel-Frequency Cepstral Coefficients (MFCC) + Deltas + Delta-Deltas
    pooled across time via (mean, std) into a fixed 78-dimensional unit vector.
    """

    @staticmethod
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
        if len(audio) < 512:
            return np.zeros(n_mfcc * 6, dtype=np.float32)

        # STFT
        frame_len = 512
        hop_len = 256
        f, t, zxx = signal.stft(audio, fs=sr, nperseg=frame_len, noverlap=frame_len - hop_len)
        mag_spec = np.abs(zxx) ** 2

        # Mel filterbank
        n_mels = 26
        low_mel = self.hz_to_mel(100.0)
        high_mel = self.hz_to_mel(sr / 2.0)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        bin_points = np.floor((frame_len + 1) * hz_points / sr).astype(int)

        filterbank = np.zeros((n_mels, frame_len // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        # Mel energies
        mel_energies = np.dot(filterbank, mag_spec)
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
        log_mel_energies = np.log(mel_energies)

        # Discrete Cosine Transform (DCT-II)
        mfccs = fft.dct(log_mel_energies, axis=0, type=2, norm="ortho")[:n_mfcc, :]

        # First and second derivatives (Delta & Delta-Delta)
        if mfccs.shape[1] > 2:
            delta1 = np.gradient(mfccs, axis=1)
            delta2 = np.gradient(delta1, axis=1)
        else:
            delta1 = np.zeros_like(mfccs)
            delta2 = np.zeros_like(mfccs)

        # Statistical pooling: mean and std along time axis
        mean_mfcc = np.mean(mfccs, axis=1)
        std_mfcc = np.std(mfccs, axis=1)
        mean_d1 = np.mean(delta1, axis=1)
        std_d1 = np.std(delta1, axis=1)
        mean_d2 = np.mean(delta2, axis=1)
        std_d2 = np.std(delta2, axis=1)

        embedding = np.concatenate([mean_mfcc, std_mfcc, mean_d1, std_d1, mean_d2, std_d2]).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding


class ECAPASpeakerAdapter(BaseSpeakerConsistency):
    """
    Adapter for ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation)
    speaker embedding extractor. Uses MFCC statistical fallback when weights are not loaded.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path or settings.ECAPA_MODEL_PATH
        self.device = device or settings.DEVICE
        self.fallback = MFCCSpeakerEmbeddingExtractor()
        self.is_mock = True

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        return self.fallback.extract_embedding(audio, sr)
