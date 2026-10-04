"""VaR exception indicators and observed exception rates."""

import numpy as np
import pandas as pd

from riskchallenger.backtesting.results import ViolationStatistics


def violation_indicators(
    realized_losses: pd.Series, var_forecasts: pd.Series
) -> pd.Series:
    """Mark strict exceedances: a violation occurs only when loss > VaR."""
    if not realized_losses.index.equals(var_forecasts.index):
        raise ValueError("realized losses and VaR forecasts must share the same index")
    violations = realized_losses > var_forecasts
    violations.name = "violation"
    return violations.astype(bool)


def calculate_violation_statistics(
    violations: pd.Series | np.ndarray, confidence_level: float
) -> ViolationStatistics:
    """Summarize observed violations against the nominal 1-alpha rate."""
    values = np.asarray(violations, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("violations must be a non-empty one-dimensional sequence")
    count = int(values.sum())
    expected_rate = 1.0 - confidence_level
    observed_rate = count / values.size
    return ViolationStatistics(
        number_of_forecasts=int(values.size),
        violations=count,
        violation_rate=float(observed_rate),
        expected_violation_rate=float(expected_rate),
        rate_difference=float(observed_rate - expected_rate),
    )
