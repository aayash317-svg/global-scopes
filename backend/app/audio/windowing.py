"""
Sliding Window Audio Segmenter
Splits continuous or streaming audio into 3.0-4.0s overlapping windows for real-time analysis.
"""

from dataclasses import dataclass
from typing import List, Generator
import numpy as np


@dataclass
class AudioWindow:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    samples: np.ndarray


class AudioWindowSegmenter:
    """
    Slices audio streams into fixed-length sliding windows with configurable overlap.
    Master reference recommendation: 3.0 to 4.0 seconds per window.
    """

    def __init__(
        self,
        window_seconds: float = 3.5,
        overlap_ratio: float = 0.5,
        min_window_seconds: float = 1.0,
    ):
        self.window_seconds = window_seconds
        self.overlap_ratio = max(0.0, min(0.9, overlap_ratio))
        self.min_window_seconds = min_window_seconds

    def slice_windows(self, audio: np.ndarray, sr: int = 16000) -> List[AudioWindow]:
        """Slices full audio array into list of AudioWindow objects."""
        return list(self.generate_windows(audio, sr))

    def generate_windows(self, audio: np.ndarray, sr: int = 16000) -> Generator[AudioWindow, None, None]:
        """Generates AudioWindow objects sequentially."""
        if len(audio) == 0:
            return

        window_size = int(self.window_seconds * sr)
        hop_size = int(window_size * (1.0 - self.overlap_ratio))
        min_size = int(self.min_window_seconds * sr)

        total_samples = len(audio)

        # If audio is shorter than window_size but exceeds min_size
        if total_samples < window_size:
            if total_samples >= min_size:
                # Pad with silence or repeat to reach window size
                pad_width = window_size - total_samples
                padded = np.pad(audio, (0, pad_width), mode="constant")
                yield AudioWindow(
                    index=0,
                    start_sec=0.0,
                    end_sec=round(total_samples / sr, 3),
                    duration_sec=round(total_samples / sr, 3),
                    samples=padded,
                )
            return

        # Slicing loop
        start_idx = 0
        window_idx = 0

        while start_idx + min_size <= total_samples:
            end_idx = min(start_idx + window_size, total_samples)
            chunk = audio[start_idx:end_idx]

            # If final chunk is slightly smaller than window_size, pad to window_size
            if len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)), mode="constant")

            yield AudioWindow(
                index=window_idx,
                start_sec=round(start_idx / sr, 3),
                end_sec=round(end_idx / sr, 3),
                duration_sec=round((end_idx - start_idx) / sr, 3),
                samples=chunk,
            )

            window_idx += 1
            start_idx += hop_size

            # Prevent endless loop if hop_size <= 0
            if hop_size <= 0:
                break
