"""
Risk Context Subsystem
Models call metadata and operational context influencing the dynamic risk score.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskContext:
    """Contextual metadata surrounding the call session."""
    scenario: str = "routine_support"  # routine_support, high_value_transaction, privileged_access
    call_origin: str = "VoIP"          # VoIP, PSTN, SIP, WebRTC, UNKNOWN
    is_known_contact: bool = True
    transaction_value: float = 0.0     # Monetary value if transaction underway
    fraud_history_score: float = 0.0   # 0.0 (clean) to 1.0 (high past fraud)
    caller_account_age_days: int = 365
    geographic_risk_weight: float = 1.0  # Multiplier for anomalous IP / telecom switch
    supervisor_escalation: bool = False

    def get_contextual_multiplier(self) -> float:
        """Computes risk multiplier derived from contextual threat indicators."""
        mult = 1.0

        # Scenario sensitivity
        if self.scenario == "privileged_access":
            mult *= 1.4
        elif self.scenario == "high_value_transaction":
            mult *= 1.25

        # Call origin risk (unverified VoIP vs authenticated PSTN)
        if self.call_origin in ("VoIP", "UNKNOWN"):
            mult *= 1.15

        # Unknown contact
        if not self.is_known_contact:
            mult *= 1.2

        # Fraud history
        if self.fraud_history_score > 0.3:
            mult *= (1.0 + self.fraud_history_score * 0.5)

        # High transaction value escalation
        if self.transaction_value > 500000.0:  # > 5 Lakh INR
            mult *= 1.3
        elif self.transaction_value > 50000.0:
            mult *= 1.15

        return round(mult, 3)
