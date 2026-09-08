"""
Risk Thresholds & Scenario Configuration
Defines decision thresholds and safety floor guarantees for different operational tiers.
"""

from dataclasses import dataclass
from typing import Dict
from config.settings import settings


@dataclass
class ScenarioThresholds:
    low_max: float      # Scores <= low_max are LOW risk
    medium_max: float   # Scores <= medium_max are MEDIUM risk
    high_max: float     # Scores <= high_max are HIGH risk; above is CRITICAL
    medium_safety_floor: float = 0.90  # Single signal >= 90% forces at least MEDIUM
    high_safety_floor: float = 0.97    # Single signal >= 97% forces at least HIGH


class RiskThresholdManager:
    """Manages risk decision boundaries across different operational scenarios."""

    DEFAULT_THRESHOLDS: Dict[str, ScenarioThresholds] = {
        "routine_support": ScenarioThresholds(
            low_max=40.0,
            medium_max=settings.RISK_ROUTINE_THRESHOLD,  # 65.0
            high_max=85.0,
        ),
        "high_value_transaction": ScenarioThresholds(
            low_max=25.0,
            medium_max=settings.RISK_HIGH_VALUE_THRESHOLD,  # 45.0
            high_max=70.0,
        ),
        "privileged_access": ScenarioThresholds(
            low_max=15.0,
            medium_max=settings.RISK_PRIVILEGED_THRESHOLD,  # 30.0
            high_max=55.0,
        ),
    }

    @classmethod
    def get_thresholds(cls, scenario: str) -> ScenarioThresholds:
        return cls.DEFAULT_THRESHOLDS.get(scenario, cls.DEFAULT_THRESHOLDS["routine_support"])
