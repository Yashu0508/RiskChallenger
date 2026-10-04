"""Common validation for risk model inputs."""

from collections.abc import Iterable

import numpy as np

from riskchallenger.risk.result import RiskUnit


def validate_confidence_level(confidence_level: float) -> float:
    """Return a valid confidence level strictly inside (0, 1)."""
    if isinstance(confidence_level, bool):
        raise ValueError("confidence_level must satisfy 0 < alpha < 1")
    try:
        alpha = float(confidence_level)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "confidence_level must be numeric and satisfy 0 < alpha < 1"
        ) from error
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("confidence_level must satisfy 0 < alpha < 1")
    return alpha


def validate_loss_observations(
    observations: Iterable[object], *, minimum_sample_size: int
) -> np.ndarray:
    """Convert one-dimensional numeric losses to finite float observations."""
    try:
        values = np.asarray(observations, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "loss observations must be numeric and one-dimensional"
        ) from error
    if values.ndim != 1:
        raise ValueError("loss observations must be one-dimensional")
    if values.size == 0:
        raise ValueError("loss observations must not be empty")
    if values.size < minimum_sample_size:
        raise ValueError(
            f"at least {minimum_sample_size} loss observation(s) are required; "
            f"received {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "loss observations must all be finite; missing values are not imputed"
        )
    return values


def validate_unit(unit: RiskUnit | str) -> RiskUnit:
    """Parse a required risk unit and reject unspecified or unknown units."""
    try:
        return unit if isinstance(unit, RiskUnit) else RiskUnit(unit)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in RiskUnit)
        raise ValueError(f"unit must be explicitly set to one of: {allowed}") from error
