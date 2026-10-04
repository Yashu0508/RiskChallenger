"""Backtesting configuration and input validation."""

from datetime import date, datetime
from numbers import Integral

import numpy as np
import pandas as pd

from riskchallenger.risk.validation import validate_confidence_level


def validate_significance_level(significance_level: float) -> float:
    """Validate a test significance level strictly inside (0, 1)."""
    try:
        level = float(significance_level)
    except (TypeError, ValueError) as error:
        raise ValueError("significance_level must satisfy 0 < level < 1") from error
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("significance_level must satisfy 0 < level < 1")
    return level


def validate_count(value: int, name: str, *, minimum: int = 1) -> int:
    """Validate a positive integer count without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def parse_optional_date(value: date | datetime | str | pd.Timestamp | None, name: str):
    """Convert an optional date boundary to a normalized pandas timestamp."""
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid date") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    return timestamp


def validate_backtest_series(series: pd.Series, name: str) -> pd.Series:
    """Require a finite numeric Series with sorted, unique datetime index."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="raise"))
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must have datetime index and numeric values"
        ) from error
    if index.hasnans:
        raise ValueError(f"{name} index contains invalid dates")
    if index.has_duplicates:
        raise ValueError(f"{name} index contains duplicate dates")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be chronological")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    result = values.copy()
    result.index = index
    result.index.name = series.index.name or "date"
    return result


def validate_confidence(confidence_level: float) -> float:
    """Use the risk package's shared alpha validation."""
    return validate_confidence_level(confidence_level)


def validate_violation_sequence(violations: object) -> np.ndarray:
    """Validate booleans or exact 0/1 values without coercing arbitrary data."""
    try:
        raw = np.asarray(violations)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "violations must be a one-dimensional boolean sequence"
        ) from error
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("violations must be a non-empty one-dimensional sequence")
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=False)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw.astype(float)).all():
        raise ValueError("violations must contain only boolean or exact 0/1 values")
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("violations must contain only boolean or exact 0/1 values")
    return raw.astype(bool)
