"""Validation for Champion–Challenger configuration and inputs."""

import math

import numpy as np


def validate_probability(value: float, name: str) -> float:
    """Validate a finite probability strictly between zero and one."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1") from error
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    return result


def validate_nonnegative(value: float, name: str) -> float:
    """Validate a finite non-negative scalar."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def validate_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    """Require finite, non-negative component weights summing to one."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("four finite non-negative scoring weights are required")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("scoring weights must sum to 1.0")
    return tuple(float(value) for value in values)
