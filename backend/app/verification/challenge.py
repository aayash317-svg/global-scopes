"""
Interactive Verification Challenge Subsystem
Generates and verifies dynamic voice challenges (numeric sequences / dynamic phrases)
to defeat pre-recorded or static cloned voice playback.
"""

import secrets
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ChallengePrompt:
    challenge_id: str
    token: str
    challenge_type: str  # "numeric_sequence", "passphrase"
    prompt_text: str
    expected_response: str
    expires_at: float
    created_at: float


class ChallengeService:
    """Manages dynamic voice challenge-response lifecycles."""

    CHALLENGE_WORDS = [
        "river", "sky", "silver", "mountain", "falcon", "beacon",
        "crimson", "orbit", "harbor", "zenith", "crystal", "horizon"
    ]

    def __init__(self, default_ttl_sec: int = 300):
        self.default_ttl_sec = default_ttl_sec
        self._active_challenges: Dict[str, ChallengePrompt] = {}

    def generate_challenge(
        self,
        session_id: str,
        user_id: str,
        challenge_type: str = "numeric_sequence",
    ) -> ChallengePrompt:
        """Generates a dynamic challenge that an attacker with a static clone cannot predict."""
        token = secrets.token_urlsafe(24)
        now = time.time()
        expires_at = now + self.default_ttl_sec

        if challenge_type == "numeric_sequence":
            # Generate 4 random digits spaced out
            digits = [str(random.randint(0, 9)) for _ in range(4)]
            expected = "".join(digits)
            prompt_text = f"Please state the following verification code clearly: {' - '.join(digits)}"
        else:
            # Generate 3 random words
            words = random.sample(self.CHALLENGE_WORDS, 3)
            expected = " ".join(words)
            prompt_text = f"Please read the following verification words: {expected}"

        challenge = ChallengePrompt(
            challenge_id=f"chal_{secrets.token_hex(8)}",
            token=token,
            challenge_type=challenge_type,
            prompt_text=prompt_text,
            expected_response=expected,
            expires_at=expires_at,
            created_at=now,
        )

        self._active_challenges[token] = challenge
        return challenge

    def verify_challenge(self, token: str, response_text: str) -> Dict[str, bool]:
        """Validates claimant response against expected prompt."""
        challenge = self._active_challenges.get(token)
        if not challenge:
            return {"valid": False, "expired": False, "reason": "Challenge token not found."}

        if time.time() > challenge.expires_at:
            del self._active_challenges[token]
            return {"valid": False, "expired": True, "reason": "Challenge token has expired."}

        # Normalize and compare
        norm_expected = challenge.expected_response.replace(" ", "").replace("-", "").lower()
        norm_actual = response_text.replace(" ", "").replace("-", "").lower()

        is_match = norm_expected == norm_actual
        if is_match:
            del self._active_challenges[token]

        return {
            "valid": is_match,
            "expired": False,
            "reason": "Challenge verified successfully." if is_match else "Spoken phrase did not match prompt.",
        }
