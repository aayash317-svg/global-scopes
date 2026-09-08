"""
Application Configuration
Voice Integrity Verification Framework (SIH Problem Statement 26104)
"""

import os
from typing import Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore


class Settings(BaseSettings):
    # Base application settings
    PROJECT_NAME: str = "Voice Integrity Verification Framework"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API configuration
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database configuration
    # Default SQLite async URL; can be overridden by DATABASE_URL env var (e.g. postgresql+asyncpg://...)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./voice_integrity.db")
    
    # Cryptography / Privacy configuration
    # 32-byte url-safe base64 key for Fernet symmetric encryption of voiceprints
    # Default key provided for dev/test; must be overridden in production via FERNET_KEY
    FERNET_KEY: str = os.getenv(
        "FERNET_KEY", 
        "U3Zvb1JvYnVzdFNlY3JldEtleUZvclZvaWNlUHJpbnQxMjM0NTY3ODk="
    )
    
    # Audio processing parameters
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_WINDOW_SECONDS: float = 3.5
    AUDIO_OVERLAP_RATIO: float = 0.5
    VAD_AGGRESSIVENESS: int = 2
    MIN_SPEECH_DURATION_SECONDS: float = 1.0
    
    # Machine Learning / Inference configuration
    DEVICE: str = os.getenv("DEVICE", "cpu")
    USE_GPU: bool = False
    AASIST_MODEL_PATH: Optional[str] = os.getenv("AASIST_MODEL_PATH", None)
    WAVLM_MODEL_PATH: Optional[str] = os.getenv("WAVLM_MODEL_PATH", None)
    ECAPA_MODEL_PATH: Optional[str] = os.getenv("ECAPA_MODEL_PATH", None)
    DIARIZATION_MODEL_PATH: Optional[str] = os.getenv("DIARIZATION_MODEL_PATH", None)
    ENABLE_MODEL_STUBS: bool = True
    
    # Risk Engine thresholds & safety floor rules (from SIH26104 Spec)
    RISK_ROUTINE_THRESHOLD: float = 65.0
    RISK_HIGH_VALUE_THRESHOLD: float = 45.0
    RISK_PRIVILEGED_THRESHOLD: float = 30.0
    SAFETY_FLOOR_MEDIUM_CONFIDENCE: float = 0.90
    SAFETY_FLOOR_HIGH_CONFIDENCE: float = 0.97
    
    # Privacy, Retention & DPDP Act 2023 Compliance
    DATA_RETENTION_DAYS: int = 30
    AUTO_DELETE_RAW_AUDIO: bool = True
    CONSENT_REQUIRED_FOR_ENROLLMENT: bool = True
    
    # Edge / Offline capabilities
    OFFLINE_MODE: bool = False
    EDGE_QUEUE_DIR: str = "./edge_queue"

    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env", 
            env_file_encoding="utf-8", 
            extra="ignore"
        )
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"


settings = Settings()
