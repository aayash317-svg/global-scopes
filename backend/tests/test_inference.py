"""
Inference Subsystem Tests
Validates acoustic, spectral, prosody, speaker consistency, replay, and unified pipeline.
"""

import pytest
import numpy as np

from backend.app.inference.deepfake_detector import (
    AcousticSpectralFeatureExtractor,
    AASISTDetectorAdapter,
    WavLMDetectorAdapter,
)
from backend.app.inference.speaker_consistency import (
    MFCCSpeakerEmbeddingExtractor,
    ECAPASpeakerAdapter,
)
from backend.app.inference.replay_detector import AcousticReplayDetector
from backend.app.inference.pipeline import UnifiedInferencePipeline


def test_acoustic_spectral_feature_extraction():
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    signal = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.random.normal(0, 0.05, len(t))).astype(np.float32)

    extractor = AcousticSpectralFeatureExtractor()
    ac = extractor.extract_acoustic_features(signal, sr)
    sp = extractor.extract_spectral_features(signal, sr)
    pr = extractor.extract_prosody_features(signal, sr)

    assert "phase_jitter" in ac
    assert "acoustic_score" in ac
    assert 0.0 <= ac["acoustic_score"] <= 1.0

    assert "spectral_flatness" in sp
    assert "spectral_score" in sp
    assert 0.0 <= sp["spectral_score"] <= 1.0

    assert "f0_std" in pr
    assert "prosody_score" in pr
    assert 0.0 <= pr["prosody_score"] <= 1.0


def test_aasist_and_wavlm_adapters():
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    signal = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    aasist = AASISTDetectorAdapter()
    res_aasist = aasist.predict(signal, sr)
    assert 0.0 <= res_aasist.spoof_probability <= 1.0
    assert res_aasist.is_mock is True  # Clearly labeled mock/stub

    wavlm = WavLMDetectorAdapter()
    res_wavlm = wavlm.predict(signal, sr)
    assert 0.0 <= res_wavlm.spoof_probability <= 1.0
    assert res_wavlm.is_mock is True


def test_speaker_consistency_embedding():
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    voice_a = (0.6 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    voice_b = (0.6 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

    adapter = ECAPASpeakerAdapter()
    emb_a1 = adapter.extract_embedding(voice_a, sr)
    emb_a2 = adapter.extract_embedding(voice_a, sr)
    emb_b = adapter.extract_embedding(voice_b, sr)

    # Identical voice should yield near 1.0 cosine similarity
    sim_same = adapter.compute_similarity(emb_a1, emb_a2)
    assert sim_same > 0.95

    # Verification workflow
    verif_res = adapter.verify(voice_a, emb_a1, threshold=0.75, sr=sr)
    assert verif_res.is_consistent is True


def test_replay_detector():
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    signal = (0.5 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

    detector = AcousticReplayDetector()
    res = detector.predict(signal, sr)
    assert 0.0 <= res.replay_probability <= 1.0
    assert "cutoff_artifact_score" in dir(res)


def test_unified_inference_pipeline():
    sr = 16000
    pipeline = UnifiedInferencePipeline()

    # Case 1: Silence chunk -> VAD gates it to insufficient speech
    silence = np.zeros(sr * 3, dtype=np.float32)
    res_silence = pipeline.process_window(silence, sr=sr)
    assert res_silence.insufficient_speech is True
    assert res_silence.has_speech is False

    # Case 2: Speech chunk with enrolled voiceprint
    t = np.linspace(0, 3, sr * 3)
    speech = (0.5 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    emb = pipeline.speaker_verifier.extract_embedding(speech, sr=sr)

    res_speech = pipeline.process_window(speech, enrolled_embedding=emb, sr=sr)
    assert res_speech.has_speech is True
    assert res_speech.speaker_similarity is not None
    assert 0.0 <= res_speech.spoof_probability <= 1.0
