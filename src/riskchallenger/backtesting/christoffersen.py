"""Christoffersen independence and conditional-coverage tests."""

import numpy as np
from scipy.stats import chi2

from riskchallenger.backtesting.kupiec import _bernoulli_log_likelihood, kupiec_test
from riskchallenger.backtesting.results import CoverageTestResult
from riskchallenger.backtesting.validation import (
    validate_significance_level,
    validate_violation_sequence,
)


def _validate_sequence(violations: np.ndarray | list[bool]) -> np.ndarray:
    sequence = validate_violation_sequence(violations)
    if sequence.size < 2:
        raise ValueError(
            "Christoffersen tests require at least two violation indicators"
        )
    return sequence


def christoffersen_independence_test(
    violations: np.ndarray | list[bool], significance_level: float = 0.05
) -> CoverageTestResult:
    """Compare iid exception likelihood with a two-state Markov likelihood."""
    sequence = _validate_sequence(violations)
    significance = validate_significance_level(significance_level)
    previous, current = sequence[:-1], sequence[1:]
    n00 = int(np.sum((~previous) & (~current)))
    n01 = int(np.sum((~previous) & current))
    n10 = int(np.sum(previous & (~current)))
    n11 = int(np.sum(previous & current))

    transitions = int(sequence.size - 1)
    total_violations = n01 + n11
    unconditional_probability = total_violations / transitions
    null_loglike = _bernoulli_log_likelihood(
        total_violations, transitions, unconditional_probability
    )

    from_zero = n00 + n01
    from_one = n10 + n11
    probability_after_zero = n01 / from_zero if from_zero else 0.0
    probability_after_one = n11 / from_one if from_one else 0.0
    markov_loglike = _bernoulli_log_likelihood(
        n01, from_zero, probability_after_zero
    ) + _bernoulli_log_likelihood(n11, from_one, probability_after_one)
    statistic = max(0.0, -2.0 * (null_loglike - markov_loglike))
    p_value = float(chi2.sf(statistic, df=1))
    counts = {"N00": n00, "N01": n01, "N10": n10, "N11": n11}
    return CoverageTestResult(
        statistic=float(statistic),
        p_value=p_value,
        degrees_of_freedom=1,
        reject_null=bool(p_value < significance),
        transition_counts=counts,
    )


def conditional_coverage_test(
    violations: np.ndarray | list[bool],
    confidence_level: float,
    significance_level: float = 0.05,
) -> CoverageTestResult:
    """Combine Kupiec unconditional coverage and Christoffersen independence."""
    sequence = _validate_sequence(violations)
    significance = validate_significance_level(significance_level)
    unconditional = kupiec_test(sequence, confidence_level, significance)
    independence = christoffersen_independence_test(sequence, significance)
    statistic = unconditional.statistic + independence.statistic
    p_value = float(chi2.sf(statistic, df=2))
    return CoverageTestResult(
        statistic=float(statistic),
        p_value=p_value,
        degrees_of_freedom=2,
        reject_null=bool(p_value < significance),
    )
