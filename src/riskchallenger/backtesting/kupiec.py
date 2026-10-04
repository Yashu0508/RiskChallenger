"""Kupiec unconditional-coverage likelihood-ratio test."""

import numpy as np
from scipy.stats import chi2

from riskchallenger.backtesting.results import CoverageTestResult
from riskchallenger.backtesting.validation import (
    validate_confidence,
    validate_significance_level,
    validate_violation_sequence,
)


def _bernoulli_log_likelihood(successes: int, total: int, probability: float) -> float:
    """Evaluate binomial log likelihood with 0*log(0) defined as zero."""
    failures = total - successes
    if (probability == 0.0 and successes > 0) or (probability == 1.0 and failures > 0):
        return -np.inf
    result = 0.0
    if successes:
        result += successes * np.log(probability)
    if failures:
        result += failures * np.log1p(-probability)
    return float(result)


def kupiec_test(
    violations: np.ndarray | list[bool],
    confidence_level: float,
    significance_level: float = 0.05,
) -> CoverageTestResult:
    """Test H0: exception probability equals 1-confidence_level."""
    alpha = validate_confidence(confidence_level)
    significance = validate_significance_level(significance_level)
    observed = validate_violation_sequence(violations)
    n = int(observed.size)
    k = int(observed.sum())
    null_probability = 1.0 - alpha
    empirical_probability = k / n
    null_loglike = _bernoulli_log_likelihood(k, n, null_probability)
    unrestricted_loglike = _bernoulli_log_likelihood(k, n, empirical_probability)
    statistic = max(0.0, -2.0 * (null_loglike - unrestricted_loglike))
    p_value = float(chi2.sf(statistic, df=1))
    return CoverageTestResult(
        statistic=float(statistic),
        p_value=p_value,
        degrees_of_freedom=1,
        reject_null=bool(p_value < significance),
    )
