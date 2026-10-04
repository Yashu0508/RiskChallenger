"""Structured outputs and units shared by risk models."""

from dataclasses import dataclass, field
from enum import Enum


class RiskUnit(str, Enum):
    """Unit class for supplied loss observations."""

    RELATIVE = "relative"
    MONETARY = "monetary"


@dataclass(frozen=True)
class RiskResult:
    """VaR and Expected Shortfall under an explicitly identified convention."""

    model_name: str
    confidence_level: float
    var: float
    expected_shortfall: float
    sample_size: int
    loss_convention: str
    unit: RiskUnit
    model_parameters: dict[str, float | int | str | None] = field(default_factory=dict)
