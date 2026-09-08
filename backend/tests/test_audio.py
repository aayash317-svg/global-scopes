"""
Audio Subsystem Tests
Validates preprocessing, VAD, windowing, quality analysis, and telephony processing.
"""

import pytest
import numpy as np
import io
import wave

from backend.app.audio.preprocessing import (
    load_audio,
    validate_audio_input,
    resample_audio,
    remove_dc_offset,
    peak_normalize,
)
from backend.app.audio.vad import VoiceActivityDetector
from backend.app.audio.windowing import AudioWindowSegmenter
from backend.app.audio.quality import AudioQualityAnalyzer
from backend.app.audio.telephony import TelephonyProcessor


def test_validate_audio_input(synthetic_wav_bytes):
    info = validate_audio_input(synthetic_wav_bytes)
    assert info["valid"] is True
    assert info["is_wav"] is True

    # Test rejection of tiny payloads
    with pytest.raises(ValueError):
        validate_audio_input(b"short")


def test_load_and_preprocess_audio(synthetic_wav_bytes):
    audio, sr = load_audio(synthetic_wav_bytes, target_sr=16000)
    assert sr == 16000
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert np.max(np.abs(audio)) <= 1.0


def test_dc_offset_and_normalize():
    arr = np.array([0.5, 0.7, 0.9, 1.1], dtype=np.float32)
    no_dc = remove_dc_offset(arr)
    assert abs(float(np.mean(no_dc))) < 1e-6

    normalized = peak_normalize(no_dc, target_peak=0.95)
    assert abs(float(np.max(np.abs(normalized))) - 0.95) < 1e-4


def test_resample_audio():
    orig_sr = 8000
    target_sr = 16000
    t = np.linspace(0, 1, orig_sr)
    s = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled = resample_audio(s, orig_sr, target_sr)
    assert len(resampled) == target_sr


def test_voice_activity_detector():
    sr = 16000
    vad = VoiceActivityDetector()

    # Pure silence
    silence = np.zeros(sr * 2, dtype=np.float32)
    vad_res = vad.detect_speech(silence, sr)
    assert vad_res["has_speech"] is False
    assert vad_res["speech_ratio"] == 0.0

    # Active tone with amplitude
    t = np.linspace(0, 2, sr * 2)
    active_speech = (0.5 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    vad_res2 = vad.detect_speech(active_speech, sr)
    assert vad_res2["has_speech"] is True
    assert vad_res2["speech_ratio"] > 0.5


def test_audio_windowing():
    sr = 16000
    audio = np.random.randn(sr * 10).astype(np.float32)
    segmenter = AudioWindowSegmenter(window_seconds=3.5, overlap_ratio=0.5)
    windows = segmenter.slice_windows(audio, sr)

    assert len(windows) > 1
    assert windows[0].duration_sec >= 3.0
    assert len(windows[0].samples) == int(3.5 * sr)


def test_audio_quality_analyzer():
    sr = 16000
    analyzer = AudioQualityAnalyzer()

    # Clean signal
    t = np.linspace(0, 3, sr * 3)
    clean = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    q = analyzer.analyze(clean, sr)
    assert q["is_usable"] is True
    assert q["clipping_ratio"] == 0.0

    # Clipped signal
    clipped = np.clip(clean * 10.0, -1.0, 1.0)
    q_clipped = analyzer.analyze(clipped, sr)
    assert q_clipped["clipping_ratio"] > 0.0


def test_telephony_processor():
    sr = 16000
    proc = TelephonyProcessor()
    t = np.linspace(0, 2, sr * 2)
    signal = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    degraded_a = proc.simulate_pstn_channel(signal, sr, codec="a_law")
    degraded_mu = proc.simulate_pstn_channel(signal, sr, codec="mu_law")

    assert len(degraded_a) == len(signal)
    assert len(degraded_mu) == len(signal)
