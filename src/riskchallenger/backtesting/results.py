"""Structured VaR backtesting outputs and coverage-test results."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CoverageTestResult:
    """Likelihood-ratio test outcome and test-specific counts."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    reject_null: bool
    transition_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class ViolationStatistics:
    """Observed versus nominal exception frequency."""

    number_of_forecasts: int
    violations: int
    violation_rate: float
    expected_violation_rate: float
    rate_difference: float


@dataclass(frozen=True)
class ModelBacktestResult:
    """Forecast series and coverage tests for one VaR model."""

    model_name: str
    confidence_level: float
    runtime_seconds: float
    forecasts: pd.DataFrame
    violation_statistics: ViolationStatistics
    kupiec: CoverageTestResult
    christoffersen: CoverageTestResult
    conditional_coverage: CoverageTestResult
    refit_frequency: int = 1


@dataclass(frozen=True)
class BacktestRunResult:
    """Results for multiple models over a common test period."""

    config: object
    model_results: dict[str, ModelBacktestResult]

    def comparison_table(self) -> pd.DataFrame:
        """Create the specified model-level test comparison table."""
        rows: list[dict[str, object]] = []
        for name, result in self.model_results.items():
            stats = result.violation_statistics
            rows.append(
                {
                    "model": name,
                    "confidence_level": result.confidence_level,
                    "runtime_seconds": result.runtime_seconds,
                    "number_of_forecasts": stats.number_of_forecasts,
                    "violations": stats.violations,
                    "violation_rate": stats.violation_rate,
                    "expected_rate": stats.expected_violation_rate,
                    "Kupiec_statistic": result.kupiec.statistic,
                    "Kupiec_p_value": result.kupiec.p_value,
                    "Christoffersen_statistic": result.christoffersen.statistic,
                    "Christoffersen_p_value": result.christoffersen.p_value,
                    "ConditionalCoverage_statistic": (
                        result.conditional_coverage.statistic
                    ),
                    "ConditionalCoverage_p_value": result.conditional_coverage.p_value,
                }
            )
        return pd.DataFrame(rows)
