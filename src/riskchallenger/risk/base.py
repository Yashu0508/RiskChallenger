"""Abstract fit/calculate interface shared by risk models."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from riskchallenger.logging import get_logger
from riskchallenger.risk.result import RiskResult, RiskUnit
from riskchallenger.risk.validation import (
    validate_confidence_level,
    validate_loss_observations,
    validate_unit,
)

LOGGER = get_logger("risk")


class RiskModel(ABC):
    """Common fitted-model interface for loss-based VaR and ES estimators."""

    model_name: str
    minimum_sample_size: int

    def __init__(self) -> None:
        self._losses: np.ndarray | None = None
        self._unit: RiskUnit | None = None

    def fit(self, losses: object, *, unit: RiskUnit | str) -> Self:
        """Validate and store finite loss observations and their explicit unit."""
        values = validate_loss_observations(
            losses, minimum_sample_size=self.minimum_sample_size
        )
        self._losses = values.copy()
        self._unit = validate_unit(unit)
        LOGGER.info(
            "Fitted %s risk model on %d %s loss observations",
            self.model_name,
            values.size,
            self._unit.value,
        )
        return self

    def calculate(self, confidence_level: float) -> RiskResult:
        """Calculate both VaR and ES at the requested confidence level."""
        if self._losses is None or self._unit is None:
            raise RuntimeError(f"{self.model_name} must be fitted before calculation")
        alpha = validate_confidence_level(confidence_level)
        var, expected_shortfall, parameters = self._calculate(alpha, self._losses)
        if not np.isfinite([var, expected_shortfall]).all():
            raise ValueError(f"{self.model_name} produced non-finite risk estimates")
        LOGGER.info(
            "Calculated %s VaR/ES at confidence %.6g",
            self.model_name,
            alpha,
        )
        return RiskResult(
            model_name=self.model_name,
            confidence_level=alpha,
            var=float(var),
            expected_shortfall=float(expected_shortfall),
            sample_size=self._result_sample_size(),
            loss_convention="positive values are losses; negative values are gains",
            unit=self._unit,
            model_parameters=parameters,
        )

    def _result_sample_size(self) -> int:
        """Size of the empirical sample underlying the reported estimate."""
        if self._losses is None:
            raise RuntimeError(f"{self.model_name} must be fitted before calculation")
        return int(self._losses.size)

    def calculate_var(self, confidence_level: float) -> float:
        """Return VaR using the common calculate operation."""
        return self.calculate(confidence_level).var

    def calculate_es(self, confidence_level: float) -> float:
        """Return Expected Shortfall using the common calculate operation."""
        return self.calculate(confidence_level).expected_shortfall

    @abstractmethod
    def _calculate(
        self, confidence_level: float, losses: np.ndarray
    ) -> tuple[float, float, dict[str, float | int | str | None]]:
        """Return VaR, ES, and model-specific fit metadata."""
