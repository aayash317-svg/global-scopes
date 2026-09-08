"""
Biometric Encryption Subsystem
Implements Layer 4 Fernet symmetric authenticated encryption for speaker embeddings.
Ensures voiceprints are encrypted at rest and decrypted strictly transiently in-memory.
"""

import base64
import numpy as np
from cryptography.fernet import Fernet
from config.settings import settings


class BiometricEncryptionService:
    """Manages encryption and decryption of biometric voiceprint embeddings."""

    def __init__(self, key: str = None):
        raw_key = key or settings.FERNET_KEY
        # Ensure 32-byte urlsafe base64 key
        try:
            # Validate if it's already a valid Fernet key
            self.fernet = Fernet(raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key)
        except Exception:
            # If invalid format, derive a valid key via base64 padding/hashing
            key_bytes = hashlib_key = raw_key.encode("utf-8")
            import hashlib
            derived = base64.urlsafe_b64encode(hashlib.sha256(key_bytes).digest())
            self.fernet = Fernet(derived)

    def encrypt_embedding(self, embedding: np.ndarray) -> str:
        """
        Serializes float32 numpy embedding array and encrypts via Fernet.
        Returns encrypted string safe for database storage.
        """
        arr_bytes = embedding.astype(np.float32).tobytes()
        encrypted_bytes = self.fernet.encrypt(arr_bytes)
        return encrypted_bytes.decode("utf-8")

    def decrypt_embedding(self, encrypted_str: str) -> np.ndarray:
        """
        Decrypts Fernet ciphertext in-memory and reconstitutes float32 embedding.
        Raw audio is never restored because only fixed-length embedding was encrypted.
        """
        encrypted_bytes = encrypted_str.encode("utf-8")
        decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
        embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)
        return np.copy(embedding)
