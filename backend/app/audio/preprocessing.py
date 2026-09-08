"""
Audio Preprocessing Subsystem
Handles audio decoding, mono downmixing, resampling to 16kHz, DC offset removal, and normalization.
Isolated from model inference.
"""

import io
import wave
import struct
import numpy as np
from typing import Tuple, Union, Dict, Any
from pathlib import Path
from scipy import signal


def validate_audio_input(audio_bytes: bytes, max_size_mb: float = 25.0) -> Dict[str, Any]:
    """Validates raw audio byte payload size and header."""
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Audio file size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB).")
    if len(audio_bytes) < 44:
        raise ValueError("Audio payload too small to be a valid audio file.")
    
    # Check RIFF header for WAV files if applicable
    is_wav = audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
    return {
        "valid": True,
        "size_bytes": len(audio_bytes),
        "is_wav": is_wav,
    }


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Removes DC bias by subtracting mean value."""
    if len(audio) == 0:
        return audio
    return audio - np.mean(audio)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak normalizes audio array to target_peak [-1.0, 1.0]."""
    if len(audio) == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        return (audio / max_val) * target_peak
    return audio


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resamples audio from orig_sr to target_sr using polyphase filtering."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)
    
    # Compute greatest common divisor for rational resampling
    import math
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def load_audio(
    source: Union[bytes, io.BytesIO, str, Path],
    target_sr: int = 16000,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Decodes audio bytes or file into 1D float32 numpy array normalized to [-1.0, 1.0].
    Channels are averaged to mono.
    Resampled to target_sr (default 16000 Hz).
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            audio_bytes = f.read()
    elif isinstance(source, io.BytesIO):
        audio_bytes = source.getvalue()
    else:
        audio_bytes = source

    validate_audio_input(audio_bytes)

    # Attempt decoding via standard library wave module
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_frames = wf.readframes(n_frames)

            # Convert bytes to numpy based on sample width
            if sampwidth == 1:
                # 8-bit unsigned
                data = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
                data = (data - 128.0) / 128.0
            elif sampwidth == 2:
                # 16-bit signed
                data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 3:
                # 24-bit signed
                # Convert 3-byte tuples to 32-bit int
                int_list = []
                for i in range(0, len(raw_frames), 3):
                    b = raw_frames[i : i + 3]
                    # sign extend
                    val = struct.unpack("<i", b + (b"\xff" if b[2] & 0x80 else b"\x00"))[0]
                    int_list.append(val)
                data = np.array(int_list, dtype=np.float32) / 8388608.0
            elif sampwidth == 4:
                # 32-bit signed or float
                try:
                    data = np.frombuffer(raw_frames, dtype=np.float32)
                except Exception:
                    data = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth} bytes.")

            # Reshape channels if multi-channel and mix to mono
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = np.mean(data, axis=1)

            orig_sr = framerate
    except (wave.Error, ValueError) as e:
        # Fallback for raw 16-bit PCM assuming 16kHz mono
        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            orig_sr = target_sr
        except Exception:
            raise ValueError(f"Could not parse audio payload: {e}")

    # Remove DC offset
    data = remove_dc_offset(data)

    # Resample to target sample rate
    if orig_sr != target_sr:
        data = resample_audio(data, orig_sr, target_sr)

    # Peak normalize
    if normalize:
        data = peak_normalize(data)

    return data.astype(np.float32), target_sr
